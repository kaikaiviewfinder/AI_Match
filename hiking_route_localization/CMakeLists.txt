cmake_minimum_required(VERSION 3.0.2)
project(hiking_route_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gpx_to_route.py
  scripts/serial_altitude_node.py
  scripts/bme280_altitude_node.py
  scripts/route_matcher_node.py
  scripts/route_visualizer_node.py
  scripts/demo_sensor_player.py
  scripts/generate_demo_route.py
  scripts/nmea_gps_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch data config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
