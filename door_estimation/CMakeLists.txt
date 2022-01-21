cmake_minimum_required(VERSION 3.0.2)
project(door_estimation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
CATKIN_DEPENDS roscpp rospy std_msgs 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
