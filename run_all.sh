#!/bin/bash
# ===================================================================
# 图寻 TU XUN 一键启动脚本
# 启动底层(ROS+传感器) + 前端(React Vite) + 播放数据包 + 打开浏览器
# 运行: bash run_all.sh
# ===================================================================
set -e

SOURCE_ROS="source /opt/ros/noetic/setup.bash"
SOURCE_WS="source /home/nb/catkin_ws/devel/setup.bash"
ROUTE_CSV="/home/nb/AI/AI_Match/AI_Data/Outdoor-1/route_table.csv"
VINS_CFG="/home/nb/AI/AI_Match/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml"
BAG_FILE="/home/nb/AI/AI_Match/AI_Data/Outdoor-1/outdoor_1.bag"
FRONTEND_DIR="/home/nb/AI/AI_Match/fortend-demonstration-website"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/pipeline"
mkdir -p "$LOG"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
err()   { echo -e "${RED}[ERR]${NC} $1"; }

# ---------- cleanup ----------
info "清理旧进程..."
pkill -f roscore        2>/dev/null || true
pkill -f vins_node      2>/dev/null || true
pkill -f route_matcher  2>/dev/null || true
pkill -f route_visualizer 2>/dev/null || true
pkill -f rosbridge      2>/dev/null || true
pkill -f http.server    2>/dev/null || true
pkill -f static_transform 2>/dev/null || true
pkill -f gps_gate       2>/dev/null || true
pkill -f mjpeg_server   2>/dev/null || true
pkill -f rosbag         2>/dev/null || true
sleep 2
ok "旧进程清理完毕"

# ---------- roscore ----------
info "1/9 启动 roscore..."
eval "$SOURCE_ROS"
roscore > "$LOG/roscore.log" 2>&1 < /dev/null &
sleep 4
rosparam set use_sim_time true
ok "roscore 就绪"

# ---------- rosbridge ----------
info "2/9 启动 rosbridge (WebSocket :9090)..."
eval "$SOURCE_ROS"
roslaunch rosbridge_server rosbridge_websocket.launch > "$LOG/rosbridge.log" 2>&1 < /dev/null &
sleep 3
ok "rosbridge :9090"

# ---------- frontend ----------
info "3/9 启动前端 (Vite :5173)..."
cd "$FRONTEND_DIR/tuxun-ui"
./node_modules/.bin/vite --port 5173 --host 0.0.0.0 > "$LOG/vite.log" 2>&1 < /dev/null &
cd "$FRONTEND_DIR"
sleep 4
ok "前端 :5173"

# ---------- VINS-Fusion ----------
info "4/9 启动 VINS-Fusion..."
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun vins vins_node "$VINS_CFG" > "$LOG/vins.log" 2>&1 < /dev/null &
sleep 3
ok "vins_estimator"

# ---------- TF bridge ----------
info "5/9 启动 TF 桥接 (map->world)..."
eval "$SOURCE_ROS"
rosrun tf static_transform_publisher 0 0 0 0 0 0 map world 100 > "$LOG/tf.log" 2>&1 < /dev/null &
sleep 1
ok "TF bridge"

# ---------- route matcher ----------
info "6/9 启动路线匹配器..."
eval "$SOURCE_ROS && $SOURCE_WS"
rosrun hiking_route_localization route_matcher_node.py \
    _route_csv:="$ROUTE_CSV" \
    _num_particles:=1500 \
    _uniform_init_without_gps:=true \
    _auto_baro_offset:=true > "$LOG/matcher.log" 2>&1 < /dev/null &
sleep 2
ok "route_matcher_node"

# ---------- GPS gate ----------
info "7/9 启动 GPS 门控(断续模拟)..."
eval "$SOURCE_ROS"
python3 "$SCRIPT_DIR/fortend-demonstration-website/gps_gate.py" > "$LOG/gps_gate.log" 2>&1 < /dev/null &
sleep 1
ok "gps_gate"

# ---------- MJPEG video ----------
info "8/9 启动视频流 (:8080)..."
eval "$SOURCE_ROS"
python3 "$SCRIPT_DIR/fortend-demonstration-website/mjpeg_server.py" > "$LOG/mjpeg.log" 2>&1 < /dev/null &
sleep 2
ok "视频流 :8080"

# ---------- open browser ----------
info "打开浏览器 http://localhost:5173/ ..."
xdg-open http://localhost:5173/ 2>/dev/null || echo "  请手动打开 http://localhost:5173/"
sleep 2

# ---------- play bag ----------
info "9/9 播放数据包 outdoor_1.bag (0.5x, GPS经由门控稀疏化)..."
eval "$SOURCE_ROS"
rosbag play "$BAG_FILE" /gnss0:=/gps_raw --clock -r 0.5 > "$LOG/bag.log" 2>&1 < /dev/null &

sleep 5

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  图寻 TU XUN — 全部启动完毕！${NC}"
echo ""
echo -e "  ${CYAN}前端页面${NC}   http://localhost:5173/"
echo -e "  ${CYAN}视频流${NC}     http://localhost:8080/stream"
echo -e "  ${CYAN}rosbridge${NC}  ws://localhost:9090"
echo -e "  ${CYAN}日志${NC}       /tmp/pipeline/"
echo ""
echo -e "  ${CYAN}GPS 模式${NC}   断续(每4-35m随机,15%长断档)"
echo -e "  ${CYAN}播放速度${NC}   0.5x (约13分钟跑完)"
echo -e "  ${CYAN}数据包${NC}     outdoor_1.bag (6m33s, 538m路线)"
echo ""
echo -e "  ${GREEN}→ 等约25秒让 VINS 初始化,页面就出数据${NC}"
echo -e "  ${GREEN}→ 停止: pkill -f roscore${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"

# keep terminal alive
wait
