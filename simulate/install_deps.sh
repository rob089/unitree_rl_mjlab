#!/usr/bin/env bash
# System dependencies required to build unitree_mujoco (simulate/).
sudo apt-get update
sudo apt-get install -y \
    libglfw3-dev \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    joystick
