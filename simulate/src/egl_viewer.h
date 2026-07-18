#pragma once

/**
 * @brief Offscreen viewer for headless machines.
 *
 * Renders the scene into an EGL pbuffer (no X server, GPU-accelerated) and
 * serves the frames over plain HTTP as PNGs. View it from a laptop with:
 *
 *     ssh -L 8080:localhost:8080 <server>
 *     # then open http://localhost:8080 in a browser
 *
 * The render thread holds the same mutex the physics loop uses, so scenes are
 * never assembled from a half-stepped mjData.
 */

#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <mujoco/mujoco.h>

#include "lodepng.h"

class EglViewer
{
public:
    EglViewer(mjModel *m, mjData *d, std::recursive_mutex &sim_mutex, int port,
              int fps, int width, int height)
        : m_(m), d_(d), sim_mutex_(sim_mutex), port_(port), fps_(fps),
          width_(width), height_(height)
    {
    }

    ~EglViewer()
    {
        running_ = false;
        if (listen_fd_ >= 0)
        {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
        }
        if (render_thread_.joinable()) render_thread_.join();
        if (http_thread_.joinable()) http_thread_.join();
    }

    bool start()
    {
        if (!initHttp())
        {
            return false;
        }
        render_thread_ = std::thread(&EglViewer::renderLoop, this);
        http_thread_ = std::thread(&EglViewer::httpLoop, this);
        std::cout << "Viewer: http://localhost:" << port_
                  << "  (ssh -L " << port_ << ":localhost:" << port_ << " <server>)"
                  << std::endl;
        return true;
    }

private:
    //---------------------------------------- rendering ----------------------------------------

    // There is no X server here, so EGL_DEFAULT_DISPLAY does not resolve. Try
    // the DRM device platform first, then Mesa's surfaceless platform, and only
    // fall back to the default display.
    bool openDisplay()
    {
        auto query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
        auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));

        if (query_devices && get_platform_display)
        {
            EGLDeviceEXT devices[8];
            EGLint num_devices = 0;
            if (query_devices(8, devices, &num_devices))
            {
                for (int i = 0; i < num_devices; ++i)
                {
                    EGLDisplay dpy =
                        get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr))
                    {
                        egl_display_ = dpy;
                        std::cout << "Viewer: EGL device " << i << " ("
                                  << eglQueryString(dpy, EGL_VENDOR) << ")" << std::endl;
                        return true;
                    }
                }
            }
        }

        if (get_platform_display)
        {
            EGLDisplay dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
                                                  EGL_DEFAULT_DISPLAY, nullptr);
            if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr))
            {
                egl_display_ = dpy;
                std::cout << "Viewer: EGL surfaceless" << std::endl;
                return true;
            }
        }

        EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, nullptr, nullptr))
        {
            egl_display_ = dpy;
            return true;
        }
        return false;
    }

    bool initEgl()
    {
        if (!openDisplay())
        {
            std::cerr << "Viewer: no usable EGL display" << std::endl;
            return false;
        }

        const EGLint config_attr[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_STENCIL_SIZE, 8,
            EGL_NONE};

        EGLConfig config;
        EGLint num_config = 0;
        if (!eglChooseConfig(egl_display_, config_attr, &config, 1, &num_config) ||
            num_config < 1)
        {
            std::cerr << "Viewer: eglChooseConfig failed" << std::endl;
            return false;
        }

        // A pbuffer is optional: MuJoCo renders into its own FBO, so a
        // surfaceless context is fine when the platform has no pbuffer support.
        const EGLint pbuffer_attr[] = {
            EGL_WIDTH, width_, EGL_HEIGHT, height_, EGL_NONE};
        egl_surface_ = eglCreatePbufferSurface(egl_display_, config, pbuffer_attr);

        if (!eglBindAPI(EGL_OPENGL_API))
        {
            std::cerr << "Viewer: eglBindAPI failed" << std::endl;
            return false;
        }

        egl_context_ = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, nullptr);
        if (egl_context_ == EGL_NO_CONTEXT)
        {
            std::cerr << "Viewer: eglCreateContext failed" << std::endl;
            return false;
        }

        return eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
    }

    void renderLoop()
    {
        if (!initEgl())
        {
            std::cerr << "Viewer: disabled (EGL unavailable)" << std::endl;
            return;
        }

        mjvScene scn;
        mjvOption opt;
        mjvCamera cam;
        mjrContext con;

        mjv_defaultScene(&scn);
        mjv_defaultOption(&opt);
        mjv_defaultCamera(&cam);
        mjr_defaultContext(&con);

        // The offscreen framebuffer defaults to the model's <visual global
        // offwidth/offheight> (640x480); size it to the requested resolution or
        // the frame comes back letterboxed.
        m_->vis.global.offwidth = width_;
        m_->vis.global.offheight = height_;

        mjv_makeScene(m_, &scn, 2000);
        mjr_makeContext(m_, &con, mjFONTSCALE_100);
        mjr_setBuffer(mjFB_OFFSCREEN, &con);
        if (con.currentBuffer != mjFB_OFFSCREEN)
        {
            std::cerr << "Viewer: offscreen buffer unavailable" << std::endl;
            return;
        }

        // Chase the robot's base so it never walks out of frame.
        int track_body = mj_name2id(m_, mjOBJ_BODY, "torso_link");
        if (track_body < 0) track_body = mj_name2id(m_, mjOBJ_BODY, "base_link");
        if (track_body >= 0)
        {
            cam.type = mjCAMERA_TRACKING;
            cam.trackbodyid = track_body;
            cam.distance = 3.5;
            cam.azimuth = 120;
            cam.elevation = -15;
        }

        const mjrRect viewport{0, 0, width_, height_};
        std::vector<unsigned char> rgb(3 * width_ * height_);
        std::vector<unsigned char> flipped(3 * width_ * height_);
        const auto frame_interval =
            std::chrono::duration<double>(1.0 / std::max(1, fps_));

        while (running_)
        {
            const auto frame_start = std::chrono::steady_clock::now();

            {
                const std::unique_lock<std::recursive_mutex> lock(sim_mutex_);
                mjv_updateScene(m_, d_, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            }
            mjr_render(viewport, &scn, &con);
            mjr_readPixels(rgb.data(), nullptr, viewport, &con);

            // OpenGL returns rows bottom-up; PNG wants top-down.
            const int row_bytes = 3 * width_;
            for (int y = 0; y < height_; ++y)
            {
                std::memcpy(&flipped[y * row_bytes],
                            &rgb[(height_ - 1 - y) * row_bytes], row_bytes);
            }

            // Default zlib settings dominate the frame budget; trade a larger
            // payload for a much cheaper encode (this is a localhost stream).
            lodepng::State state;
            state.info_raw.colortype = LCT_RGB;
            state.info_png.color.colortype = LCT_RGB;
            state.encoder.zlibsettings.btype = 2;
            state.encoder.zlibsettings.windowsize = 512;
            state.encoder.zlibsettings.minmatch = 3;
            state.encoder.zlibsettings.nicematch = 16;

            std::vector<unsigned char> png;
            if (lodepng::encode(png, flipped, width_, height_, state) == 0)
            {
                const std::lock_guard<std::mutex> lock(frame_mutex_);
                frame_.swap(png);
            }

            std::this_thread::sleep_until(frame_start + frame_interval);
        }

        mjr_freeContext(&con);
        mjv_freeScene(&scn);
    }

    //------------------------------------------ http -------------------------------------------

    bool initHttp()
    {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return false;

        int reuse = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // ssh tunnel only
        addr.sin_port = htons(port_);

        if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
            ::listen(listen_fd_, 4) < 0)
        {
            std::cerr << "Viewer: cannot bind port " << port_ << std::endl;
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }
        return true;
    }

    void httpLoop()
    {
        while (running_)
        {
            const int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0)
            {
                if (!running_) break;
                continue;
            }

            char req[1024] = {0};
            const ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
            if (n > 0)
            {
                if (std::strstr(req, "GET /frame.png"))
                {
                    std::vector<unsigned char> png;
                    {
                        const std::lock_guard<std::mutex> lock(frame_mutex_);
                        png = frame_;
                    }
                    sendResponse(fd, "image/png", png.data(), png.size());
                }
                else
                {
                    const std::string page = indexPage();
                    sendResponse(fd, "text/html", page.data(), page.size());
                }
            }
            ::close(fd);
        }
    }

    void sendResponse(int fd, const char *type, const void *body, std::size_t len)
    {
        const std::string header =
            "HTTP/1.1 200 OK\r\nContent-Type: " + std::string(type) +
            "\r\nContent-Length: " + std::to_string(len) +
            "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
        ::send(fd, header.data(), header.size(), MSG_NOSIGNAL);
        if (len) ::send(fd, body, len, MSG_NOSIGNAL);
    }

    std::string indexPage() const
    {
        // Poll frame.png rather than using multipart streaming: simpler, and it
        // degrades gracefully if the browser stalls.
        return R"(<!doctype html><html><head><title>unitree_mujoco (headless)</title>
<style>body{margin:0;background:#111;display:flex;align-items:center;
justify-content:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img id="v"><script>
const img=document.getElementById('v');
function next(){const n=new Image();n.onload=()=>{img.src=n.src;setTimeout(next,)" +
               std::to_string(1000 / std::max(1, fps_)) +
               R"();};n.onerror=()=>setTimeout(next,500);n.src='/frame.png?t='+Date.now();}
next();</script></body></html>)";
    }

    mjModel *m_;
    mjData *d_;
    std::recursive_mutex &sim_mutex_;
    const int port_, fps_, width_, height_;

    std::atomic<bool> running_{true};
    std::thread render_thread_, http_thread_;
    int listen_fd_{-1};

    std::mutex frame_mutex_;
    std::vector<unsigned char> frame_;

    EGLDisplay egl_display_{EGL_NO_DISPLAY};
    EGLSurface egl_surface_{EGL_NO_SURFACE};
    EGLContext egl_context_{EGL_NO_CONTEXT};
};
