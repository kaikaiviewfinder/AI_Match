#!/bin/bash
# ===================================================================
# 图寻 TU XUN 一键关闭脚本
# 终止所有底层(ROS+传感器) + 前端(React Vite) + rosbag 进程
# 双击即可运行，支持中文输出
# ===================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}════════════════════════════════════════════════════${NC}"
echo -e "${RED}  图寻 TU XUN — 一键关闭所有进程${NC}"
echo -e "${RED}════════════════════════════════════════════════════${NC}"
echo ""

killed_count=0

# 按端口杀（最可靠，覆盖所有监听该端口的进程）
kill_port() {
    local port=$1
    local desc=$2
    local k=0
    for pid in $(lsof -ti :$port 2>/dev/null); do
        kill -9 $pid 2>/dev/null && k=$((k+1))
    done
    [ $k -gt 0 ] && echo -e "  ${GREEN}端口 :${port}${NC} → 已终止 ${k} 个进程 (${desc})" && killed_count=$((killed_count+k))
}

# 按关键字杀
kill_proc() {
    local pattern=$1
    local desc=$2
    local k=0
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
        # 跳过本脚本自身
        [ "$pid" = "$$" ] && continue
        kill -9 $pid 2>/dev/null && k=$((k+1))
    done
    [ $k -gt 0 ] && echo -e "  ${GREEN}${desc}${NC} (${pattern}) → 已终止 ${k} 个进程" && killed_count=$((killed_count+k))
}

# ---- 1) 先杀端口（最快最准） ----

echo -e "${CYAN}[1/3]${NC} 按端口清理..."
kill_port 8080  "视频流 mjpeg"
kill_port 9090  "rosbridge websocket"
kill_port 5173  "前端 Vite"
kill_port 11311 "ROS master"

# ---- 2) 按关键字杀剩余进程 ----

echo -e "${CYAN}[2/3]${NC} 按进程名清理..."

kill_proc "rosbag.*play"              "rosbag 播放器"
kill_proc "vins_node"                 "VINS-Fusion"
kill_proc "route_matcher_node"        "路线匹配器"
kill_proc "route_visualizer"          "路线可视化"
kill_proc "rosbridge_websocket"       "rosbridge 节点"
kill_proc "rosapi"                    "rosapi 节点"
kill_proc "rosmaster"                 "rosmaster"
kill_proc "roscore"                   "roscore"
kill_proc "rosout"                    "rosout"
kill_proc "gps_gate"                  "GPS 门控"
kill_proc "mjpeg_server"              "MJPEG 服务"
kill_proc "static_transform_publisher" "TF 桥接"
kill_proc "roslaunch"                 "roslaunch"
kill_proc "node_modules/.bin/vite"   "Vite 前端"

# ---- 3) 最终兜底：杀所有 ROS/Python 相关 ----
echo -e "${CYAN}[3/3]${NC} 兜底清理..."
# 杀掉所有属于当前用户的 ros 相关 Python 进程
for pid in $(ps -u "$USER" -o pid,comm --no-headers 2>/dev/null | grep -E "(ros|rosbag|rosmaster|rosout|rosbridge)" | awk '{print $1}'); do
    [ "$pid" = "$$" ] && continue
    kill -9 $pid 2>/dev/null && killed_count=$((killed_count+1))
done

# 清理 /tmp/pipeline 日志（可选，保留以便排查）
# rm -rf /tmp/pipeline/*.log 2>/dev/null

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  完成！已清理所有进程${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"

# 验证：检查常见端口是否已释放
echo ""
echo -e "${CYAN}验证端口释放状态...${NC}"
all_clear=true
for port in 8080 9090 5173 11311; do
    if lsof -ti :$port >/dev/null 2>&1; then
        echo -e "  ${RED}:${port} ⚠ 仍被占用${NC}"
        all_clear=false
    else
        echo -e "  ${GREEN}:${port} ✓ 已释放${NC}"
    fi
done
$all_clear && echo -e "\n${GREEN}所有端口已释放，可以重新启动。${NC}"

exit 0
