#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    // Headless viewer (unitree_mujoco_headless only)
    int viewer_port = 0; // 0 disables the offscreen viewer
    int start_paused = 0; // begin frozen so the robot does not fall while you connect
    std::filesystem::path init_qpos; // optional file with a qpos vector used on reset
    int viewer_fps = 20;
    int viewer_width = 960;
    int viewer_height = 540;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("viewer_port,p", po::value<int>(&config.viewer_port), "Headless viewer HTTP port, 0=off; -p 8080")
        ("init_qpos", po::value<std::filesystem::path>(&config.init_qpos),
            "File with a whitespace-separated qpos vector applied on reset")
        ("paused", po::bool_switch()->notifier([](bool v){ if(v) config.start_paused = 1; }),
            "Start with physics paused; press 'g' to run")
        ("viewer_fps", po::value<int>(&config.viewer_fps), "Headless viewer frame rate")
        ("viewer_width", po::value<int>(&config.viewer_width), "Headless viewer width")
        ("viewer_height", po::value<int>(&config.viewer_height), "Headless viewer height")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}