cmake_minimum_required(VERSION 3.0.2)
project(sim_code_v2)

## Compile options
add_compile_options(-O3 -DNDEBUG)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  tf
  cv_bridge
  image_transport
)

## System dependencies
find_package(Boost REQUIRED COMPONENTS system)

## Declare this project as a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs nav_msgs geometry_msgs sensor_msgs tf cv_bridge image_transport
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Python scripts
catkin_install_python(PROGRAMS
  nodes/simple_controller.py
  nodes/teleop_keyboard.py
  nodes/lane_follower_bev.py
  nodes/m3_gate_nav.py
  nodes/m4_cone_follower.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
