cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
# Prefer the active conda env's libraries (fmt in particular: unitree_sdk2's
# spdlog needs fmt 12, which is newer than the system package on Ubuntu 24.04).
if (DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt CONFIG REQUIRED)


include_directories(
  mujoco/include
  mujoco/simulate
  src/lodepng
)
link_directories(mujoco/lib)

file(GLOB SIM_SRC
  src/joystick/joystick.cc

  # exclude mujoco/simulate/main.cc
  mujoco/simulate/glfw_*.cc
  mujoco/simulate/platform_*.cc
  mujoco/simulate/simulate.cc
  src/lodepng/lodepng.cpp
)

# Headless simulator: physics + DDS bridge only, no GLFW/rendering.
find_package(OpenGL REQUIRED COMPONENTS EGL)

add_executable(unitree_mujoco_headless
  src/headless_main.cc
  src/joystick/joystick.cc
  src/lodepng/lodepng.cpp
)
target_link_libraries(unitree_mujoco_headless
  pthread
  mujoco
  yaml-cpp
  unitree_sdk2
  Boost::program_options
  fmt::fmt
  OpenGL::EGL
)
# unitree_sdk2's bundled spdlog is built against fmt 12; the system libfmt may be
# older, so pick up the versioned one CMake resolved and bake in its rpath.
set_target_properties(unitree_mujoco_headless PROPERTIES
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/lib"
  INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/mujoco/lib"
)

link_libraries(
  pthread
  mujoco
  glfw
  yaml-cpp
  unitree_sdk2
  boost_program_options
  fmt
)

# GUI build needs GLFW headers/libs; skip it on display-less machines.
find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h)
find_library(GLFW_LIBRARY NAMES glfw glfw3)
if (GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
  add_executable(unitree_mujoco ${SIM_SRC} src/main.cc)
  add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
else()
  message(STATUS "GLFW not found - building headless target only")
endif()