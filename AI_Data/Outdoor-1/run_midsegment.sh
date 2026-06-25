#!/bin/bash
# 一键启动：从路线中途录像定位自己位置
# 用法: bash run_midsegment.sh

set -e
SOURCE_ROS="source /opt/ros/noetic/setup.bash"
SOURCE_WS="source ~/catkin_ws/devel/setup.bash"

echo "=== 1. 清理旧进程 ==="
pkill -f roscore 2>/dev/null || true
pkill -f vins_node 2>/dev/null || true
pkill -f route_matcher 2>/dev/null || true
pkill -f route_visualizer 2>/dev/null || true
pkill -f rviz 2>/dev/null || true
sleep 2

echo "=== 2. 启动 roscore ==="
eval "$SOURCE_ROS"
roscore &
sleep 3

echo "=== 3. 启动 VINS-Fusion ==="
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml /odometry:=/vins_estimator/odometry &
sleep 2

echo "=== 4. 启动路线匹配器 ==="
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun hiking_route_localization route_matcher_node.py \
    _route_csv:=/home/kai/AI_Data/Outdoor-1/route_table.csv \
    _num_particles:=1500 \
    _sigma_gps:=12.0 \
    _weight_gps:=0.8 \
    _auto_baro_offset:=true &
sleep 2

echo "=== 5. 启动路线可视化 ==="
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun hiking_route_localization route_visualizer_node.py \
    _route_csv:=/home/kai/AI_Data/Outdoor-1/route_table.csv &
sleep 2

echo "=== 6. 启动 RViz ==="
LIBGL_ALWAYS_SOFTWARE=1 rviz -d ~/catkin_ws/src/hiking_route_localization/config/hiking_rviz.rviz &
sleep 4

echo "=== 7. 播放中途视频 (180s-300s, 即路线中段~322m处开始) ==="
eval "$SOURCE_ROS"
rosbag play /home/kai/AI_Data/Outdoor-1/outdoor_1.bag \
    /gnss0:=/gps --clock -r 1 --start=180 --duration=120

echo ""
echo "=== 完成！==="
echo "RViz 中应该显示:"
echo "  灰线 = 完整路线"
echo "  橙线 = 本段视频实际走过的路线 (从 ~322m 处开始，不是从0)"
echo "  红球 = 当前匹配位置"
