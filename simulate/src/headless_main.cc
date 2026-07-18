// Headless variant of unitree_mujoco: physics + unitree_sdk2 DDS bridge, no
// GLFW/rendering. Intended for running trained policies on a display-less
// server. Same DDS surface as the GUI build, so deploy/ controllers connect
// unchanged.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <unistd.h>

#include <mujoco/mujoco.h>

#include "egl_viewer.h"
#include "param.h"
#include "unitree_sdk2_bridge.h"

#define NUM_MOTOR_IDL_GO 20

namespace
{
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // Guards mjData against the viewer's scene updates. The DDS bridge is left
  // unsynchronised, matching upstream's GUI build.
  std::recursive_mutex sim_mutex;

  std::atomic<bool> exit_request{false};

  using Clock = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;

  void SignalHandler(int)
  {
    exit_request.store(true);
  }

  // Optional override of the reset pose, for testing a policy from the initial
  // state its reference motion assumes rather than from a neutral stand.
  std::vector<double> init_qpos;

  void LoadInitQpos(const std::filesystem::path &path, int nq)
  {
    std::ifstream f(path);
    if (!f)
    {
      std::cerr << "init_qpos: cannot open " << path << std::endl;
      return;
    }
    double v;
    while (f >> v) init_qpos.push_back(v);
    if (static_cast<int>(init_qpos.size()) != nq)
    {
      std::cerr << "init_qpos: expected " << nq << " values, got "
                << init_qpos.size() << " - ignoring" << std::endl;
      init_qpos.clear();
      return;
    }
    std::cout << "init_qpos: reset pose loaded from " << path << std::endl;
  }

  void ApplyReset(mjModel *m, mjData *d)
  {
    mj_resetData(m, d);
    if (!init_qpos.empty())
    {
      mju_copy(d->qpos, init_qpos.data(), m->nq);
    }
    mj_forward(m, d);
  }

  // mirrors main.cc: resolve executable directory to locate config.yaml
  std::string getExecutableDir()
  {
    char buf[4096];
    ssize_t written = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (written <= 0)
    {
      return "";
    }
    buf[written] = '\0';
    std::string realpath(buf);
    return realpath.substr(0, realpath.find_last_of('/'));
  }
} // namespace

void UnitreeSdk2BridgeThread()
{
  while (!d && !exit_request.load())
  {
    usleep(500000);
  }
  if (exit_request.load())
  {
    return;
  }
  std::cout << "Mujoco data is prepared" << std::endl;

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id,
                                                   param::config.interface);

  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0)
  {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;

  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO)
  {
    interface = std::make_unique<G1Bridge>(m, d);
  }
  else
  {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();

  while (!exit_request.load())
  {
    sleep(1);
  }
}

int main(int argc, char **argv)
{
  std::printf("MuJoCo version %s (headless)\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::filesystem::path proj_dir =
      std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if (param::config.robot_scene.is_relative())
  {
    param::config.robot_scene = proj_dir.parent_path() / param::config.robot_scene;
  }

  char load_error[1024] = "";
  m = mj_loadXML(param::config.robot_scene.c_str(), nullptr, load_error,
                 sizeof(load_error));
  if (!m)
  {
    std::printf("Failed to load %s: %s\n", param::config.robot_scene.c_str(),
                load_error);
    return 1;
  }
  d = mj_makeData(m);
  if (!param::config.init_qpos.empty())
  {
    LoadInitQpos(param::config.init_qpos, m->nq);
  }
  ApplyReset(m, d);

  std::cout << "Scene: " << param::config.robot_scene << std::endl;
  std::cout << "Timestep: " << m->opt.timestep << " s, actuators: " << m->nu
            << std::endl;

  if (param::config.start_paused == 1)
  {
    sim_keys.paused.store(true);
    std::cout << "Physics paused. Start the controller, then press 'g' to run."
              << std::endl;
  }

  std::thread bridge_thread(UnitreeSdk2BridgeThread);

  std::unique_ptr<EglViewer> viewer;
  if (param::config.viewer_port > 0)
  {
    viewer = std::make_unique<EglViewer>(
        m, d, sim_mutex, param::config.viewer_port, param::config.viewer_fps,
        param::config.viewer_width, param::config.viewer_height);
    viewer->start();
  }

  // Real-time paced stepping: keep sim time aligned with wall-clock, re-syncing
  // whenever we drift too far (e.g. after a stall).
  const double kMaxDrift = 0.1;
  auto sync_cpu = Clock::now();
  double sync_sim = d->time;

  while (!exit_request.load())
  {
    {
      const std::unique_lock<std::recursive_mutex> lock(sim_mutex);

      if (sim_keys.reset_requested.exchange(false))
      {
        ApplyReset(m, d);
        sync_cpu = Clock::now();
        sync_sim = d->time;
      }

      if (sim_keys.paused.load())
      {
        mj_forward(m, d); // keep sensors/DDS state fresh while frozen
      }
      else
      {
        mj_step(m, d);
      }
    }

    if (sim_keys.paused.load())
    {
      // Hold real time still too, so resuming does not trigger a catch-up burst.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      sync_cpu = Clock::now();
      sync_sim = d->time;
      continue;
    }

    const double elapsed_sim = d->time - sync_sim;
    const double elapsed_cpu = Seconds(Clock::now() - sync_cpu).count();
    const double ahead = elapsed_sim - elapsed_cpu;

    if (ahead > 0)
    {
      std::this_thread::sleep_for(Seconds(ahead));
    }
    else if (-ahead > kMaxDrift)
    {
      // Running behind real time; drop the deficit instead of sprinting.
      sync_cpu = Clock::now();
      sync_sim = d->time;
    }
  }

  // Leave teardown to process exit: the DDS bridge thread may still be touching
  // mj_data_ from a subscriber callback, so freeing it here would race.
  std::cout << "\nShutting down..." << std::endl;
  std::cout.flush();
  bridge_thread.detach();
  _exit(0);
}
