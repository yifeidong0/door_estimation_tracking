#=============================================================================
# Copyright (C) 2021, Robotic Systems Lab, ETH Zurich
# All rights reserved.
# http://www.rsl.ethz.ch
# https://bitbucket.org/leggedrobotics/mobile_manipulation
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
cmake_minimum_required(VERSION 3.0.2)
project(door_estimation_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  message_generation
  std_msgs
)

#==
# Declare ROS messages, services and actions
#==

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DoorEstimationResult.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

#==
# catkin specific configuration
#==

catkin_package(
  CATKIN_DEPENDS
    sensor_msgs
    message_runtime
    std_msgs
)

#==
# Install
#==

install(DIRECTORY
  msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
