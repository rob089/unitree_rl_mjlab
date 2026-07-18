#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <termios.h>
#include <thread>
#include <unistd.h>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

/**
 * @brief Simulator-level keys, polled by the physics loop.
 *
 * Separate from the gamepad emulation below: these control the simulation
 * itself, not the robot.
 */
struct SimKeyState
{
    std::atomic<bool> paused{false};
    std::atomic<bool> reset_requested{false};
};
inline SimKeyState sim_keys;

/**
 * @brief Virtual gamepad driven by the keyboard, for headless/SSH use.
 *
 * Reads raw keystrokes from stdin and presents them through the same
 * UnitreeJoystick interface a physical pad uses, so the deploy-side FSM sees
 * ordinary button combos (e.g. "LT + up.on_pressed").
 *
 * Button combos are latched for kHoldTime so that the FSM's .on_pressed edge
 * detection fires reliably; the axes decay back to zero once key repeat stops.
 */
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    KeyboardJoystick()
    {
        if (!isatty(fileno(stdin)))
        {
            // No terminal (nohup, systemd, piped input): stay neutral instead of
            // spinning on EOF.
            std::cout << "KeyboardJoystick: stdin is not a terminal, input disabled."
                      << std::endl;
            return;
        }
        has_tty_ = true;
        tcgetattr(fileno(stdin), &old_settings_);
        termios raw = old_settings_;
        raw.c_lflag &= (~ICANON & ~ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1; // 100 ms read timeout, so the thread can exit
        tcsetattr(fileno(stdin), TCSANOW, &raw);

        printHelp();
        read_thread_ = std::thread(&KeyboardJoystick::readLoop, this);
    }

    ~KeyboardJoystick()
    {
        running_ = false;
        if (read_thread_.joinable())
        {
            read_thread_.join();
        }
        if (has_tty_)
        {
            tcsetattr(fileno(stdin), TCSANOW, &old_settings_);
        }
    }

    static void printHelp()
    {
        std::cout << "\n=== Keyboard gamepad ===\n"
                  << "  1  FixStand   (LT + up)\n"
                  << "  2  Velocity   (RT + A)\n"
                  << "  0  Passive    (LT + B)\n"
                  << "  7  Punch      (RB + X)\n"
                  << "  8  Kick       (RB + Y)\n"
                  << "  9  Wave       (RB + B)\n"
                  << "  w/s  forward / backward     a/d  strafe left / right\n"
                  << "  q/e  turn left / right      space  stop\n"
                  << "  g  pause / resume physics   r  reset robot\n"
                  << "  h  show this help           Ctrl-C  quit\n"
                  << "Hold a movement key (key repeat) to keep the command alive.\n\n";
    }

    // Called from the bridge thread at the LowState publish rate.
    void update() override
    {
        const auto now = Clock::now();
        const auto since_press = now - combo_time_.load();
        const Combo combo = combo_.load();

        // LT/RT are smoothed axes: they need ~20 updates to cross their
        // threshold, whereas the action button's .on_pressed edge fires on the
        // first update. Hold the modifier down first, then press the action, or
        // expressions like "LT + up.on_pressed" never evaluate true.
        const bool modifier_active = since_press < kHoldTime;

        // The deploy FSM polls the joystick more slowly than LowState arrives, so
        // a single .on_pressed edge is usually missed. Pulse the action button
        // for the rest of the window: one of the edges always lands on a poll.
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(since_press).count();
        const bool action_active =
            since_press >= kModifierLead && since_press < kHoldTime &&
            ((ms / kPulseMs) % 2 == 0);

        LT(modifier_active && (combo == Combo::kFixStand || combo == Combo::kPassive) ? 1.0f : 0.0f);
        RT(modifier_active && combo == Combo::kVelocity ? 1.0f : 0.0f);
        RB(modifier_active && (combo == Combo::kPunch || combo == Combo::kKick ||
                               combo == Combo::kWave)
               ? 1
               : 0);
        up(action_active && combo == Combo::kFixStand ? 1 : 0);
        A(action_active && combo == Combo::kVelocity ? 1 : 0);
        B(action_active && (combo == Combo::kPassive || combo == Combo::kWave) ? 1 : 0);
        X(action_active && combo == Combo::kPunch ? 1 : 0);
        Y(action_active && combo == Combo::kKick ? 1 : 0);

        // Axes hold their value while the key keeps repeating, then decay.
        const bool axis_active = (now - axis_time_.load()) < kAxisHoldTime;
        if (!axis_active)
        {
            ly_cmd_ = 0.0f;
            lx_cmd_ = 0.0f;
            rx_cmd_ = 0.0f;
        }
        ly(ly_cmd_.load());
        lx(lx_cmd_.load());
        rx(rx_cmd_.load());
        ry(0.0f);

        // Unused on this layout, but keep the base class state consistent.
        back(0);
        start(0);
        LB(0);
        down(0);
        left(0);
        right(0);
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr auto kHoldTime = std::chrono::milliseconds(1500);
    static constexpr auto kModifierLead = std::chrono::milliseconds(250);
    static constexpr long kPulseMs = 60; // action button on/off period
    static constexpr auto kAxisHoldTime = std::chrono::milliseconds(400);

    enum class Combo
    {
        kNone,
        kFixStand,
        kVelocity,
        kPassive,
        kPunch,
        kKick,
        kWave
    };

    void triggerCombo(Combo c)
    {
        combo_.store(c);
        combo_time_.store(Clock::now());
    }

    void setAxes(float ly_v, float lx_v, float rx_v)
    {
        ly_cmd_.store(ly_v);
        lx_cmd_.store(lx_v);
        rx_cmd_.store(rx_v);
        axis_time_.store(Clock::now());
    }

    void readLoop()
    {
        while (running_)
        {
            char c = 0;
            if (::read(fileno(stdin), &c, 1) != 1)
            {
                continue;
            }

            switch (c)
            {
            case '1': triggerCombo(Combo::kFixStand); break;
            case '2': triggerCombo(Combo::kVelocity); break;
            case '0': triggerCombo(Combo::kPassive);  break;
            case '7': triggerCombo(Combo::kPunch);    break;
            case '8': triggerCombo(Combo::kKick);     break;
            case '9': triggerCombo(Combo::kWave);     break;

            case 'w': setAxes(kAxisStep, 0.0f, 0.0f);  break;
            case 's': setAxes(-kAxisStep, 0.0f, 0.0f); break;
            case 'a': setAxes(0.0f, -kAxisStep, 0.0f); break;
            case 'd': setAxes(0.0f, kAxisStep, 0.0f);  break;
            case 'q': setAxes(0.0f, 0.0f, -kAxisStep); break;
            case 'e': setAxes(0.0f, 0.0f, kAxisStep);  break;
            case ' ': setAxes(0.0f, 0.0f, 0.0f);       break;

            case 'g':
            {
                const bool paused = !sim_keys.paused.load();
                sim_keys.paused.store(paused);
                std::cout << (paused ? "[paused]\n" : "[running]\n") << std::flush;
                break;
            }
            case 'r':
                sim_keys.reset_requested.store(true);
                std::cout << "[reset]\n" << std::flush;
                break;

            case 'h': printHelp(); break;
            default: break;
            }
        }
    }

    static constexpr float kAxisStep = 1.0f;

    termios old_settings_{};
    bool has_tty_{false};
    std::thread read_thread_;
    std::atomic<bool> running_{true};

    std::atomic<Combo> combo_{Combo::kNone};
    std::atomic<Clock::time_point> combo_time_{Clock::time_point{}};
    std::atomic<Clock::time_point> axis_time_{Clock::time_point{}};
    std::atomic<float> lx_cmd_{0.0f}, ly_cmd_{0.0f}, rx_cmd_{0.0f};
};
