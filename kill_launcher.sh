#!/bin/bash
# 终端启动器：在终端中运行 kill_all.sh 以便看到输出
gnome-terminal -- bash -c 'bash /home/nb/AI/AI_Match/kill_all.sh; echo ""; read -p "按 Enter 关闭..."' 2>/dev/null || \
xterm -e 'bash /home/nb/AI/AI_Match/kill_all.sh; echo ""; read -p "按 Enter 关闭..."' 2>/dev/null || \
bash /home/nb/AI/AI_Match/kill_all.sh
