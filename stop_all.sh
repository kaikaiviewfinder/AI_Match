#!/bin/bash
# 停止图寻 TU XUN 所有进程
echo "停止所有进程..."
for p in rosbag route_matcher route_visualizer vins_node gps_gate mjpeg_server static_transform rosbridge "node_modules/.bin/vite" vite http.server roscore rosmaster; do
  pkill -f "$p" 2>/dev/null && echo "  已停 $p" || true
done
echo "完成。"
