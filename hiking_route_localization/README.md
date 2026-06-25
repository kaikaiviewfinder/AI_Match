# hiking_route_localization

ROS Noetic package for route-constrained hiking localization.

It fuses:

- VINS-Fusion odometry: `/vins_estimator/odometry` (`nav_msgs/Odometry`)
- intermittent GPS: `/gps` (`sensor_msgs/NavSatFix`)
- STM32/barometer altitude: `/baro_altitude` (`std_msgs/Float64`)
- known 3D hiking route: `route_table.csv`

It outputs:

- `/route_matcher/state` (`std_msgs/String`, JSON)
- `/route_matcher/matched_odometry` (`nav_msgs/Odometry`)
- `/route_matcher/progress` (`std_msgs/Float64`)
- `/route_matcher/markers` (`visualization_msgs/MarkerArray`)

## 1. Install

```bash
cd ~/catkin_ws/src
unzip hiking_route_localization.zip
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 2. Run demo without hardware

```bash
roslaunch hiking_route_localization demo_route_matcher.launch
```

Check output:

```bash
rostopic echo /route_matcher/state
```

Open RViz manually and add:

- MarkerArray: `/route_matcher/markers`
- Odometry: `/route_matcher/matched_odometry`
- Odometry: `/vins_estimator/odometry`

Or start RViz with the launch argument:

```bash
roslaunch hiking_route_localization demo_route_matcher.launch rviz:=true
```

## 3. Convert your GPX route

```bash
rosrun hiking_route_localization gpx_to_route.py your_route.gpx --output ~/route_table.csv --step 1.0
```

The GPX should contain track points with latitude/longitude and preferably elevation.

## 4. Run with real VINS-Fusion

Terminal 1, run VINS-Fusion as usual:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch vins vins_rviz.launch
```

Terminal 2:

```bash
source ~/catkin_ws/devel/setup.bash
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml
```

Terminal 3, run route matcher:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch hiking_route_localization real_route_matcher.launch route_csv:=/home/kekaiwu/route_table.csv serial_port:=/dev/ttyUSB0 serial_baud:=115200
```

If barometer serial is not ready, disable it:

```bash
roslaunch hiking_route_localization real_route_matcher.launch route_csv:=/home/kekaiwu/route_table.csv start_serial_altitude:=false
```

## 5. STM32 serial format

`serial_altitude_node.py` accepts these line formats:

```text
1719040123456,100436.37,23.79,421.82
100436.37,23.79,421.82
421.82
```

It uses the last number in each line as altitude in meters.

## 6. Topic requirements

If your GPS topic is not `/gps`, remap it:

```bash
rosrun your_gps_driver gps_node /your_gps_topic:=/gps
```

If your VINS topic is not `/vins_estimator/odometry`, remap it:

```bash
rosrun hiking_route_localization route_matcher_node.py /vins_estimator/odometry:=/your_vio_odom
```

## 7. Important parameters

Tune inside `launch/real_route_matcher.launch`:

- `sigma_gps`: GPS horizontal noise, meters
- `sigma_alt`: altitude noise, meters
- `sigma_motion`: VINS step uncertainty, meters
- `weight_gps`: GPS confidence
- `weight_alt`: altitude confidence
- `weight_slope`: altitude trend confidence
- `weight_motion`: VINS continuity confidence
- `auto_baro_offset`: automatically align first useful barometer altitude to route altitude

## 8. Limitation

This package estimates the route progress `s`, not full visual SLAM mapping. VINS-Fusion provides local motion increments; GPS and route altitude constrain the result to the known route. For real experiments, camera-IMU calibration and timestamps are still critical.
