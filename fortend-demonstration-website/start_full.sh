#!/bin/bash
# ============================================================
#  图寻 TU XUN — 完整启动脚本
#  启动: rosbridge + HTTP 前端 + VINS + 路线匹配 + RViz
# ============================================================
set -e

SOURCE_ROS="source /opt/ros/noetic/setup.bash"
SOURCE_WS="source /home/kai/catkin_ws/devel/setup.bash"
ROUTE_CSV="/home/kai/AI/AI_Data/Outdoor-1/route_table.csv"
VINS_CONFIG="/home/kai/AI/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml"
RViz_CONFIG="/home/kai/AI/hiking_route_localization/config/hiking_rviz.rviz"
FRONTEND_DIR="/home/kai/AI/fortend-demonstration-website"

# Cleanup
echo "=== 清理旧进程 ==="
pkill -f roscore 2>/dev/null || true
pkill -f vins_node 2>/dev/null || true
pkill -f route_matcher 2>/dev/null || true
pkill -f route_visualizer 2>/dev/null || true
pkill -f rosbridge 2>/dev/null || true
pkill -f rviz 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true
pkill -f static_transform 2>/dev/null || true
sleep 2

# Terminal 1: roscore
echo "=== 1/7 启动 roscore ==="
eval "$SOURCE_ROS"
roscore &
sleep 3

rosparam set use_sim_time true

# Terminal 2: rosbridge WebSocket
echo "=== 2/7 启动 rosbridge (WebSocket :9090) ==="
eval "$SOURCE_ROS"
roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 2

# Terminal 3: HTTP frontend server
echo "=== 3/7 启动前端 HTTP 服务 (:8000) ==="
cd "$FRONTEND_DIR"
python3 -m http.server 8000 &
sleep 1

# Terminal 4: VINS-Fusion
echo "=== 4/7 启动 VINS-Fusion ==="
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun vins vins_node "$VINS_CONFIG" &
sleep 3

# Terminal 5: TF bridge
echo "=== 5/7 启动 TF 桥接 ==="
eval "$SOURCE_ROS"
rosrun tf static_transform_publisher 0 0 0 0 0 0 map world 100 &
sleep 1

# Terminal 6: Route matcher + visualizer
echo "=== 6/7 启动路线匹配器 ==="
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun hiking_route_localization route_matcher_node.py \
    _route_csv:="$ROUTE_CSV" \
    _num_particles:=1500 \
    _uniform_init_without_gps:=true \
    _auto_baro_offset:=true &
sleep 2

eval "$SOURCE_ROS && $SOURCE_WS"
rosrun hiking_route_localization route_visualizer_node.py \
    _route_csv:="$ROUTE_CSV" &
sleep 1

# Terminal 7: RViz
echo "=== 7/7 启动 RViz ==="
eval "$SOURCE_ROS"
DISPLAY=:0 rviz -d "$RViz_CONFIG" &
sleep 3

echo ""
echo "============================================"
echo "  全部启动完毕！"
echo ""
echo "  前端页面: http://localhost:8000"
echo "  rosbridge: ws://localhost:9090"
echo "  VINS:      /vins_estimator/odometry"
echo "  路线匹配:  /route_matcher/state"
echo ""
echo "  播放 bag 来提供数据:"
echo "  source /opt/ros/noetic/setup.bash"
echo "  rosbag play /home/kai/AI/AI_Data/Outdoor-1/outdoor_1.bag \\"
echo "      /gnss0:=/gps --clock -r 1"
echo "============================================"

wait
