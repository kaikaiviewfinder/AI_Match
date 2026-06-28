# 徒步路线定位系统 — 技术文档

## 1. 项目概述

本项目实现了一个**基于已知路线的多传感器融合定位系统**，专门用于徒步/登山场景。核心功能：给
定一条已知的徒步路线（GPX 格式），通过手机录制视频（单目摄像头+IMU+GPS+BME280 气压计），
系统能自动识别用户在路线上的当前位置，即使录制是从路线的任意位置开始。

### 1.1 解决的问题

徒步场景中 GPS 信号微弱且断续，单独依赖 GPS 无法稳定定位。本系统融合四种传感器，利用已知
路线作为约束，实现可靠定位：

- **摄像头 + IMU** → 视觉惯性里程计（VINS-Fusion），提供连续的运动增量
- **间歇 GPS** → 提供绝对经纬度锚点，纠正累积漂移
- **BME280 气压计** → 提供连续海拔读数，匹配路线海拔剖面
- **已知路线** → 作为空间约束，将定位问题降维为一维路线进度估计

### 1.2 核心算法

**粒子滤波器 (Particle Filter)**，状态为一维路线进度 `s`（以米为单位，从路线起点算起）。

- 1200~1500 个粒子沿路线分布
- VINS 里程计驱动粒子沿路线移动（预测步）
- GPS、气压计、路线约束更新粒子权重（更新步）
- 有效粒子数过低时进行重采样

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    硬件层                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ 单目相机  │  │   IMU    │  │   GPS    │  │BME280│ │
│  │ 1280x720 │  │ 6-axis   │  │ 断续1Hz  │  │气压计│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘ │
│       │             │            │           │      │
│       ▼             ▼            ▼           ▼      │
│  ┌────────────────────────────────────────────────┐  │
│  │              rosbag 录制 (ROS)                   │  │
│  │  /cam0/image_raw  /imu0  /gnss0  /baro_altitude │  │
│  └────────────────────┬───────────────────────────┘  │
└───────────────────────┼──────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────┐
│                    软件层                              │
│                       ▼                               │
│  ┌─────────────────────────────────────┐             │
│  │         VINS-Fusion (C++)            │             │
│  │  视觉-惯性紧耦合非线性优化            │             │
│  │  输出: /vins_estimator/odometry      │             │
│  └────────────────┬────────────────────┘             │
│                   ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │    hiking_route_localization (Python)│             │
│  │                                      │             │
│  │  ┌───────────────────────────────┐  │             │
│  │  │   route_matcher_node.py       │  │             │
│  │  │   粒子滤波器 (1200-1500粒子)   │  │             │
│  │  │   输入:                        │  │             │
│  │  │   - /vins_estimator/odometry   │  │             │
│  │  │   - /gps                       │  │             │
│  │  │   - /baro_altitude             │  │             │
│  │  │   - route_table.csv            │  │             │
│  │  │   输出:                        │  │             │
│  │  │   - /route_matcher/state       │  │             │
│  │  │   - /route_matcher/odometry    │  │             │
│  │  │   - /route_matcher/progress    │  │             │
│  │  └───────────────────────────────┘  │             │
│  │                                      │             │
│  │  ┌───────────────────────────────┐  │             │
│  │  │   route_visualizer_node.py    │  │             │
│  │  │   RViz 可视化                  │  │             │
│  │  │   输出: /route_matcher/markers │  │             │
│  │  └───────────────────────────────┘  │             │
│  │                                      │             │
│  │  ┌───────────────────────────────┐  │             │
│  │  │   海拔节点 (二选一):           │  │             │
│  │  │   - serial_altitude_node.py    │  │             │
│  │  │   - bme280_altitude_node.py    │  │             │
│  │  │   输出: /baro_altitude          │  │             │
│  │  └───────────────────────────────┘  │             │
│  │                                      │             │
│  │  ┌───────────────────────────────┐  │             │
│  │  │   辅助工具:                     │  │             │
│  │  │   - gpx_to_route.py            │  │             │
│  │  │   - nmea_gps_node.py           │  │             │
│  │  │   - demo_sensor_player.py      │  │             │
│  │  │   - generate_demo_route.py     │  │             │
│  │  │   - route_tools.py (库)        │  │             │
│  │  └───────────────────────────────┘  │             │
│  └─────────────────────────────────────┘             │
│                                                       │
│  ┌─────────────────────────────────────┐             │
│  │         可视化 (RViz)                │             │
│  │   - 灰色线: 完整已知路线             │             │
│  │   - 橙色线: 本段实际走过的路段       │             │
│  │   - 红色球: 当前匹配位置             │             │
│  │   - 白色文字: 进度/置信度/剩余距离   │             │
│  │   - 绿色线: VINS 原始里程计轨迹      │             │
│  │   - 红色线: 路线匹配后轨迹            │             │
│  └─────────────────────────────────────┘             │
└───────────────────────────────────────────────────────┘
```

### 2.1 数据流

```
rosbag 播放 → /cam0/image_raw → VINS-Fusion → /vins_estimator/odometry ─┐
            → /imu0            → VINS-Fusion (IMU 预积分)                │
            → /gnss0           → remap /gps ─────────────────────────────┤
                                                                         │
                                                                         ▼
                                                     route_matcher_node.py
                                                     (粒子滤波器)
                                                            │
                                              ┌─────────────┼─────────────┐
                                              ▼             ▼             ▼
                                     /route_matcher  /route_matcher  /route_matcher
                                     /state          /matched_odom   /progress
                                     (JSON状态)      (NavSatFix)     (Float64)
```

---

## 3. 目录结构

```
/home/kai/AI/
├── VINS-Fusion/                     # 视觉惯性里程计 (HKUST开源)
│   ├── vins_estimator/              # VINS 核心估计器
│   │   ├── CMakeLists.txt
│   │   ├── package.xml              # 包名: vins
│   │   └── src/
│   │       ├── estimator/           # 状态估计、特征管理
│   │       ├── factor/              # IMU/投影残差因子
│   │       ├── featureTracker/      # KLT 光流特征追踪
│   │       ├── initial/             # 视觉-惯性初始化
│   │       └── utility/             # 可视化、工具函数
│   ├── camera_models/               # 相机模型库 (Pinhole/MEI/Cata等)
│   │   ├── CMakeLists.txt
│   │   ├── package.xml              # 包名: camera_models
│   │   └── src/camera_models/
│   ├── loop_fusion/                 # 回环检测与位姿图优化
│   │   ├── CMakeLists.txt
│   │   └── src/
│   ├── global_fusion/               # GPS 全局融合
│   │   ├── CMakeLists.txt
│   │   └── src/
│   ├── config/
│   │   ├── hiking/                  # 徒步场景配置
│   │   │   ├── hiking_mono_imu_config.yaml
│   │   │   └── cam0_pinhole.yaml
│   │   ├── euroc/                   # EuRoC 数据集配置
│   │   ├── phone/                   # 手机配置
│   │   └── ...
│   ├── docker/                      # Docker 支持
│   └── support_files/
│
├── hiking_route_localization/       # 路线匹配定位 (自研)
│   ├── CMakeLists.txt
│   ├── package.xml                  # 包名: hiking_route_localization
│   ├── scripts/
│   │   ├── route_matcher_node.py    # ★ 粒子滤波路线匹配器
│   │   ├── route_visualizer_node.py # RViz 路线可视化
│   │   ├── route_tools.py           # 路线表加载/插值/坐标转换
│   │   ├── serial_altitude_node.py  # STM32 通用串口海拔
│   │   ├── bme280_altitude_node.py  # BME280 气压计海拔
│   │   ├── nmea_gps_node.py         # NMEA GPS 串口读取
│   │   ├── gpx_to_route.py          # GPX→CSV 路线转换
│   │   ├── generate_demo_route.py   # 生成演示路线
│   │   └── demo_sensor_player.py    # 演示传感器模拟器
│   ├── launch/
│   │   ├── real_route_matcher.launch    # 真机启动
│   │   ├── demo_route_matcher.launch    # Demo 启动
│   │   └── hiking_simulation.launch
│   ├── config/
│   │   └── hiking_rviz.rviz         # RViz 可视化配置
│   └── data/
│       └── demo_route.csv           # 演示路线数据
│
├── Mobile-GVIO-calib/               # 手机相机-IMU 标定参数
│   ├── cam0_pinhole.yaml            # 相机内参
│   ├── vins.yaml                    # VINS 配置参考
│   └── orbslam3.yaml                # ORB-SLAM3 配置参考
│
├── AI_Data/                         # 测试数据
│   ├── Indoor-1/
│   │   ├── indoor_1.bag             # 室内录制 rosbag
│   │   └── ground_truth.txt
│   └── Outdoor-1/
│       ├── outdoor_1.bag            # 户外徒步 rosbag (10.1GB, 6分33秒)
│       ├── route_table.csv          # 从 GNSS 提取的路线 CSV
│       ├── ground_truth.txt
│       ├── generate_route.py        # 从 rosbag 提取路线
│       └── run_midsegment.sh        # 中途启动脚本
│
└── TECHNICAL_DOCUMENTATION.md       # 本文档
```

---

## 4. 依赖与先决条件

### 4.1 操作系统

- Ubuntu 20.04 LTS (Focal Fossa)
- ROS Noetic Ninjemys (完整桌面版)

### 4.2 系统包

```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full

# 编译工具链
sudo apt install build-essential cmake

# Ceres Solver (非线性优化)
sudo apt install libceres-dev          # 版本 >= 1.14

# Eigen3 (线性代数)
sudo apt install libeigen3-dev         # 版本 >= 3.3

# OpenCV (计算机视觉)
sudo apt install libopencv-dev         # 版本 >= 4.2

# Boost (C++ 工具库)
sudo apt install libboost-filesystem-dev libboost-program-options-dev libboost-system-dev
```

### 4.3 Python 依赖

```bash
# ROS Python (随 ROS 安装)
# rospy, std_msgs, sensor_msgs, nav_msgs, geometry_msgs, visualization_msgs, tf

# 串口通信 (用于 BME280/GPS 串口)
sudo apt install python3-serial

# 可选: 坐标转换
# GeographicLib (已在 VINS-Fusion/global_fusion 中内置)
```

### 4.4 Catkin Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace

# 链接项目包
ln -s /home/kai/AI/VINS-Fusion ~/catkin_ws/src/VINS-Fusion
ln -s /home/kai/AI/hiking_route_localization ~/catkin_ws/src/hiking_route_localization
```

---

## 5. 构建

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

构建顺序（自动按拓扑排序）：
1. `hiking_route_localization` (纯 Python，安装脚本到 devel)
2. `camera_models` (C++ 共享库 + Calibrations 工具)
3. `global_fusion` (GPS 全局融合节点)
4. `loop_fusion` (回环检测节点)
5. `vins` (VINS 主节点 + 共享库)

构建产物：
```
~/catkin_ws/devel/lib/
├── libcamera_models.so
├── libvins_lib.so
├── liblibGeographiccc.so
├── camera_models/Calibrations       # 相机标定工具
├── vins/
│   ├── vins_node                    # VINS 主程序
│   ├── kitti_odom_test
│   └── kitti_gps_test
├── loop_fusion/loop_fusion_node     # 回环检测
├── global_fusion/global_fusion_node # GPS 融合
└── hiking_route_localization/       # Python 脚本(符号链接)
    ├── route_matcher_node.py
    ├── route_visualizer_node.py
    ├── serial_altitude_node.py
    ├── bme280_altitude_node.py
    ├── nmea_gps_node.py
    ├── gpx_to_route.py
    └── ...
```

---

## 6. 传感器硬件配置

### 6.1 手机传感器 (录制 rosbag)

| 传感器 | ROS Topic | 消息类型 | 频率 | 说明 |
|--------|-----------|----------|------|------|
| 单目相机 | `/cam0/image_raw` | `sensor_msgs/Image` | ~30Hz | 1280×720 灰度或彩色 |
| IMU | `/imu0` | `sensor_msgs/Imu` | ~100Hz | 6 轴 (加速度+角速度) |
| GPS | `/gnss0` | `sensor_msgs/NavSatFix` | ~1Hz | WGS84 经纬度+海拔 |
| BME280 | `/baro_altitude` | `std_msgs/Float64` | ~20Hz | 通过串口从 STM32 读取 |

### 6.2 BME280 连接（两种方案）

**方案 A — STM32 通用串口 + serial_altitude_node.py**

STM32 串口输出行格式（任一即可）：
```
1719040123456,100436.37,23.79,421.82    # 时间戳,气压Pa,温度°C,海拔m
100436.37,23.79,421.82                  # 气压Pa,温度°C,海拔m
421.82                                  # 仅海拔m
```
脚本取最后一个数字作为海拔（米）。

**方案 B — BME280 专用 + bme280_altitude_node.py**

STM32 串口输出固定格式：
```
Temperature: 25.37 C  |  Humidity: 65.00 %  |  Pressure: 100123.00 Pa
```
脚本自动解析温度、湿度、气压，使用 ISA 标准大气模型将气压转换为海拔：
```
altitude = (T0/0.0065) × (1 - (P/P0)^0.190263)
```
其中 T0 = 温度 + 273.15 (开尔文)，P0 = 海平面参考气压 (默认 1013.25 hPa)

### 6.3 GPS 串口

通过 `nmea_gps_node.py` 读取标准 NMEA 语句（$GPGGA/$GPRMC/$GNGGA/$GNRMC），
检查校验和后发布 `/gps`。

---

## 7. 路线准备

### 7.1 GPX 文件转换

如果你的路线是 GPX 格式（包含 `<trkpt>` 或 `<rtept>` 点）：

```bash
source ~/catkin_ws/devel/setup.bash
rosrun hiking_route_localization gpx_to_route.py your_route.gpx \
    --output ~/route_table.csv \
    --step 1.0         # 重采样步长(米), 默认1m
```

GPX 文件要求：
- 包含 `<trkpt>` 或 `<rtept>` 元素
- 每个点有 `lat` 和 `lon` 属性
- 可选 `<ele>` 子元素（海拔）
- 至少 2 个点

### 7.2 从 rosbag 提取路线

如果已有录制的 rosbag（含 GNSS 数据）：

```bash
source ~/catkin_ws/devel/setup.bash
# 使用 AI_Data/Outdoor-1/generate_route.py 作为模板
python3 generate_route.py
```

### 7.3 route_table.csv 格式

```csv
type,s,x,y,z,yaw,slope,curvature,segment_id,lon,lat
origin,,,,,,,,,113.9354742,22.53983989
point,0.000,0.000,0.000,13.026,1.48608388,0.62933114,0.00000000,0,113.935474200,22.539839890
point,2.000,0.071,0.833,13.552,1.33822851,1.02441696,0.66970815,1,113.935474888,22.539847375
...
```

| 列名 | 说明 |
|------|------|
| `type` | `origin` 行定义坐标系原点(经纬度)；`point` 行为路线采样点 |
| `s` | 路线里程(米)，从起点算起的累计距离 |
| `x, y` | 局部 ENU 坐标(米)，以 origin 为原点，x=东，y=北 |
| `z` | 路线海拔(米) |
| `yaw` | 路线切线方向角(弧度)，-π~π |
| `slope` | 路线坡度，Δz/Δxy |
| `curvature` | 路线曲率，Δyaw/Δs |
| `segment_id` | 段索引 |
| `lon, lat` | WGS84 经纬度 |

### 7.4 坐标系统

- **GPS**：WGS84 经纬度 → 本地 ENU 平面近似 (WGS84 椭球，原点为路线起点)
- **局部 ENU**：x 轴指向东，y 轴指向北，z 轴指向上
- **VINS 世界系**：`world` 框架 (VINS 初始化时的局部坐标系)
- **路线系**：`map` 框架 (路线匹配器输出)
- **TF 桥接**：`static_transform_publisher 0 0 0 0 0 0 map world 100` (identity 变换)

> 注意：对于短距离徒步路线 (< 10km)，局部切平面近似足够。长距离需替换为 GeographicLib。

---

## 8. 相机-IMU 标定

### 8.1 相机内参 (cam0_pinhole.yaml)

```yaml
model_type: PINHOLE
camera_name: camera
image_width: 1280
image_height: 720
distortion_parameters:
   k1: 0.06513753258976974
   k2: -0.23698184370758457
   p1: 0.0007685090700979932
   p2: 0.005825777305547826
projection_parameters:
   fx: 996.4948765107108
   fy: 995.8706613911386
   cx: 651.7912725410048
   cy: 364.86680915422824
```

### 8.2 相机-IMU 外参 (hiking_mono_imu_config.yaml)

`body_T_cam0` 矩阵 (IMU 坐标系到相机坐标系的变换)：
```
 0.01916709  -0.99980408  -0.00494212   0.03133239
-0.99955275  -0.01904832  -0.02305337   0.00009802
 0.02295471   0.00538177  -0.99972202   0.00177312
 0.0          0.0          0.0          1.0
```

### 8.3 获取标定参数

使用 Kalibr 或 ORB-SLAM3 标定工具获取。参考文件位于 `Mobile-GVIO-calib/`。

---

## 9. 运行系统

### 9.1 Demo 模式（无需硬件，首次测试用）

使用模拟传感器数据测试系统：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch hiking_route_localization demo_route_matcher.launch rviz:=true
```

这会启动：
- `demo_sensor_player.py` — 沿 demo 路线模拟传感器 (VINS/GPS/气压)
- `route_matcher_node.py` — 粒子滤波路线匹配
- `route_visualizer_node.py` — RViz 可视化
- RViz — 图形界面

### 9.2 真机模式（完整 7 个终端）

**终端 1 — roscore**
```bash
source /opt/ros/noetic/setup.bash
roscore
```

**终端 2 — VINS-Fusion**
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
rosrun vins vins_node /home/kai/AI/VINS-Fusion/config/hiking/hiking_mono_imu_config.yaml
```
等待输出 "waiting for image and imu..." 表示就绪。

**终端 3 — TF 坐标桥接**
```bash
source /opt/ros/noetic/setup.bash
rosrun tf static_transform_publisher 0 0 0 0 0 0 map world 100
```
桥接 VINS `world` 坐标系和路线 `map` 坐标系。

**终端 4 — 路线匹配器 + 海拔传感器 + 可视化**

使用 launch 文件一键启动：
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# 使用 BME280 气压计
roslaunch hiking_route_localization real_route_matcher.launch \
    route_csv:=/home/kai/AI/AI_Data/Outdoor-1/route_table.csv \
    start_serial_altitude:=false \
    start_bme280:=true \
    bme280_port:=/dev/ttyUSB0 \
    sea_level_hpa:=1013.25

# 使用通用串口海拔
roslaunch hiking_route_localization real_route_matcher.launch \
    route_csv:=/path/to/your_route.csv \
    start_serial_altitude:=true \
    serial_port:=/dev/ttyUSB0 \
    serial_baud:=115200

# 无气压计 (仅 GPS+VINS)
roslaunch hiking_route_localization real_route_matcher.launch \
    route_csv:=/path/to/your_route.csv \
    start_serial_altitude:=false
```

**终端 5 — RViz 可视化**
```bash
source /opt/ros/noetic/setup.bash
rviz -d /home/kai/AI/hiking_route_localization/config/hiking_rviz.rviz
```

**终端 6 — 实拍视频（可选）**
```bash
source /opt/ros/noetic/setup.bash
rosrun image_view image_view image:=/cam0/image_raw _image_transport:=raw
```

**终端 7 — 播放 rosbag（录播模式）或实时录制**
```bash
source /opt/ros/noetic/setup.bash
rosparam set use_sim_time true
rosbag play /path/to/your_recording.bag \
    /gnss0:=/gps --clock -r 1
```

> 注意：`/gnss0:=/gps` 将 bag 中的 GNSS topic 重映射为路线匹配器期望的 `/gps`

### 9.3 播放中途片段（模拟"不知道从哪开始"）

```bash
# 从 180 秒开始播放 120 秒 — 模拟从路线中途开始录制
rosbag play outdoor_1.bag /gnss0:=/gps --clock --start=180 --duration=120
```

### 9.4 实时录制模式

如果手机端实时发布 ROS topics，无需 rosbag 播放：

1. 手机连接 WiFi，配置 `ROS_MASTER_URI` 指向笔记本电脑
2. 启动终端 1-5 如上
3. 手机端启动传感器驱动发布 topics

---

## 10. 配置参数详解

### 10.1 VINS-Fusion 配置 (hiking_mono_imu_config.yaml)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `imu` | 1 | 是否使用 IMU |
| `num_of_cam` | 1 | 摄像头数量 |
| `imu_topic` | "/imu0" | IMU 订阅 topic |
| `image0_topic` | "/cam0/image_raw" | 图像订阅 topic |
| `output_path` | "/home/kai/output" | 结果输出路径 |
| `cam0_calib` | "cam0_pinhole.yaml" | 相机内参文件(相对于config目录) |
| `image_width` | 1280 | 图像宽度 |
| `image_height` | 720 | 图像高度 |
| `estimate_extrinsic` | 0 | 是否在线优化外参 |
| `body_T_cam0` | (矩阵) | IMU→相机外参 |
| `acc_n` | 0.1 | 加速度计噪声密度 |
| `gyr_n` | 0.01 | 陀螺仪噪声密度 |
| `acc_w` | 0.001 | 加速度计随机游走 |
| `gyr_w` | 0.0001 | 陀螺仪随机游走 |
| `g_norm` | 9.80655 | 重力加速度 |
| `estimate_td` | 1 | 是否在线估计时间偏移 |
| `td` | 0.00181 | 初始时间偏移(秒) |
| `max_cnt` | 150 | 最大特征点数 |
| `min_dist` | 30 | 特征点最小间距(像素) |
| `freq` | 10 | 特征追踪频率(Hz) |
| `show_track` | 1 | 显示特征追踪窗口(1=显示) |
| `max_solver_time` | 0.04 | 最大求解时间(秒) |
| `max_num_iterations` | 8 | 最大优化迭代次数 |

### 10.2 路线匹配器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `route_csv` | - | 路线 CSV 文件路径 (必填) |
| `num_particles` | 1500 | 粒子数量 |
| `uniform_init_without_gps` | true | 无GPS时均匀初始化(未知起点场景) |
| `initial_s` | 0.0 | 初始路线进度(米) |
| `initial_std_m` | 80.0 | 初始粒子散布标准差(米) |
| `sigma_gps` | 12.0 | GPS 水平噪声(米) |
| `sigma_alt` | 5.0 | 海拔噪声(米) |
| `sigma_slope` | 2.0 | 坡度匹配噪声(米/米) |
| `sigma_motion` | 1.5 | 运动连续性噪声(米) |
| `motion_noise` | 0.5 | 运动预测噪声(米) |
| `gps_gate_m` | 80.0 | GPS 软门限距离(米) |
| `resample_neff_ratio` | 0.55 | 有效粒子比例阈值(低于此值重采样) |
| `max_step_m` | 5.0 | 单步最大步长(米) |
| `allow_backward` | false | 是否允许后退 |
| `weight_gps` | 0.8 | GPS 置信度权重 |
| `weight_alt` | 1.0 | 海拔置信度权重 |
| `weight_slope` | 0.6 | 海拔趋势(坡度)置信度权重 |
| `weight_motion` | 0.9 | 运动连续性权重 |
| `auto_baro_offset` | true | 自动校准气压计偏置(需GPS先锚定) |
| `baro_offset` | null | 手动设置气压计偏置(米) |
| `baro_history_window` | 8.0 | 气压历史窗口(秒),用于坡度估计 |

### 10.3 调参指南

| 场景 | 调整建议 |
|------|----------|
| GPS 信号很好 | `weight_gps=1.2`, `sigma_gps=6.0` |
| GPS 信号弱/断续 | `weight_gps=0.5`, `sigma_gps=20.0`, `gps_gate_m=100` |
| 气压计精度高 | `weight_alt=1.5`, `sigma_alt=2.0` |
| 路线坡度变化大 | `weight_slope=1.0` |
| VINS 漂移大 | `weight_motion=0.5`, `motion_noise=1.0` |
| 确信起点位置 | `uniform_init_without_gps=false`, `initial_std_m=20` |
| 从路线任意点开始 | `uniform_init_without_gps=true`, `initial_std_m=80` |

---

## 11. 输出 Topics

### 11.1 路线匹配器输出

**`/route_matcher/state`** (`std_msgs/String`, JSON)
```json
{
  "s_m": 46.212,              // 路线进度(米)
  "route_length_m": 537.914,  // 路线总长(米)
  "progress": 0.08591,        // 进度百分比 0~1
  "segment_id": 23,           // 当前路段编号
  "distance_to_end_m": 491.7, // 距终点距离(米)
  "matched_x": -1.335,        // 匹配局部坐标 X(米)
  "matched_y": -29.227,       // 匹配局部坐标 Y(米)
  "matched_z": 12.506,        // 匹配海拔(米)
  "matched_lon": 113.935461,  // 匹配经度
  "matched_lat": 22.539577,   // 匹配纬度
  "yaw_rad": -1.080783,       // 路线方向角(弧度)
  "s_std_m": 1.247,           // 定位标准差(米)
  "confidence": 0.984,        // 置信度 0~1
  "last_vio_delta_m": 0.141,  // 最近一步VIO增量(米)
  "baro_altitude_m": null,    // 气压计海拔(null=无数据)
  "altitude_error_m": null,   // 海拔误差(null=无数据)
  "gps_age_s": -43.693        // 最近GPS距现在的时间(秒)
}
```

**`/route_matcher/matched_odometry`** (`nav_msgs/Odometry`)
路线约束后的里程计信息，frame_id="map"。

**`/route_matcher/progress`** (`std_msgs/Float64`)
路线进度 0~1。

**`/route_matcher/markers`** (`visualization_msgs/MarkerArray`)
RViz 可视化标记：
- `known_route` — 完整路线 (灰线)
- `matched_position` — 当前匹配位置 (红球)
- `matched_progress` — 已走路线 (橙线)
- `route_text` — 进度/置信度/剩余距离文字

### 11.2 VINS-Fusion 输出

| Topic | 类型 | 说明 |
|-------|------|------|
| `/vins_estimator/odometry` | `nav_msgs/Odometry` | VIO 里程计 (frame_id="world") |
| `/vins_estimator/image_track` | `sensor_msgs/Image` | 特征追踪可视化 |
| `/vins_estimator/path` | `nav_msgs/Path` | 完整轨迹 |
| `/vins_estimator/keyframe_pose` | `geometry_msgs/PoseStamped` | 关键帧位姿 |
| `/vins_estimator/camera_pose` | `geometry_msgs/PoseStamped` | 相机位姿 |

---

## 12. RViz 可视化设置

使用预配置的 RViz 文件 `config/hiking_rviz.rviz`：

| 显示项 | Topic | 说明 |
|--------|-------|------|
| `Grid` | — | 参考网格 |
| `Route Markers` | `/route_matcher/markers` | 路线 + 位置 + 进度 |
| `Matched Odometry` | `/route_matcher/matched_odometry` | 红色匹配轨迹 |
| `VIO Odometry` | `/vins_estimator/odometry` | 绿色 VIO 原始轨迹 |

> 注意：如果 VIO 轨迹不显示，检查终端 3 的 TF 是否已启动，或在 RViz 中将 `Fixed Frame` 改为 `world`。

---

## 13. 关键算法说明

### 13.1 粒子滤波器

**状态定义**：一维路线进度 `s ∈ [0, L]`，其中 L 为路线总长。

**预测步 (Predict)**：每个粒子 `s_i` 根据 VINS 里程计增量 `Δd` 更新：
```
s_i = clamp(s_i + Δd + noise, 0, L)
noise ~ N(0, motion_noise²)
```

**权重更新 (Update)**：对数似然累加，四种观测：

1. **GPS 似然**（水平位置）：
   ```
   cost_gps = min(dist_xy² / (2·σ_gps²), gate² / (2·σ_gps²))
   log_weight -= weight_gps × cost_gps
   ```

2. **气压计似然**（海拔）：
   ```
   cost_alt = (route_z(s) - baro_z)² / (2·σ_alt²)
   log_weight -= weight_alt × cost_alt
   ```

3. **坡度似然**（海拔变化趋势）：
   ```
   cost_slope = (Δz_route - Δz_baro)² / (2·σ_slope²)
   log_weight -= weight_slope × cost_slope
   ```

4. **运动似然**（连续性约束）：
   ```
   cost_motion = (s - s_expected)² / (2·σ_motion²)
   log_weight -= weight_motion × cost_motion
   ```

**重采样**：当 `1/Σw² < neff_ratio × N` 时触发系统重采样。

**GPS 锚定**：GPS 首次到达时，搜索最近路线点初始化粒子。如果粒子当前位置与 GPS 偏差超过 80 米，自动触发重新锚定。

**气压计校准**：GPS 锚定后，自动校准气压计偏置：
```
baro_offset = route_z(s_estimated) - baro_z_raw
```

### 13.2 GPS 软门限

为防止 GPS 严重失准时杀死所有粒子，GPS 代价上限为门限距离内的最大代价。
门限外的粒子获得相同的最大惩罚，不至于权重为零。

### 13.3 坐标转换

本地切平面近似：WGS84 经纬度 → 以路线起点为原点的 ENU 坐标。
```
x = R × Δlon × cos(φ_mid)
y = R × Δlat
```
其中 R = 6378137m (WGS84 长半轴)。

---

## 14. 完整复现步骤

### 新设备部署清单

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install ros-noetic-desktop-full libceres-dev libeigen3-dev \
    libopencv-dev libboost-filesystem-dev libboost-program-options-dev \
    libboost-system-dev python3-serial

# 2. 克隆代码
cd ~
mkdir -p AI
# 将项目代码拷贝到 ~/AI/

# 3. 创建 catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
ln -s ~/AI/VINS-Fusion .
ln -s ~/AI/hiking_route_localization .

# 4. 编译
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 5. 准备路线
source devel/setup.bash
rosrun hiking_route_localization gpx_to_route.py your_route.gpx \
    --output ~/route_table.csv --step 1.0

# 6. 验证 demo
roslaunch hiking_route_localization demo_route_matcher.launch rviz:=true

# 7. 真机运行 (见第9节)
```

### 新设备需要替换的配置

| 文件 | 需替换内容 |
|------|-----------|
| `config/hiking/cam0_pinhole.yaml` | 你的相机内参 |
| `config/hiking/hiking_mono_imu_config.yaml` | `body_T_cam0` 外参矩阵、IMU 噪声参数 |
| `route_table.csv` | 你的徒步路线 |

---

## 15. 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| VINS 轨迹线不显示 | world/map 坐标系不匹配 | 启动 TF bridge 或将 RViz Fixed Frame 改为 world |
| 置信度始终低 | GPS/气压计质量差 | 增大 `gps_gate_m`、降低 `weight_gps` |
| 路线中间开始，定位错误 | 粒子未均匀初始化 | 设置 `uniform_init_without_gps:=true` |
| 气压计数据为空 | 串口未连接或格式不匹配 | 检查 `start_bme280` 或 `start_serial_altitude` 配置 |
| 编译报 camera_models 重复 | 嵌套目录被重复发现 | 清理 `src/camera_models` 符号链接 |
| "Connection refused" | roscore 未在同一终端启动 | 全部终端先 `source /opt/ros/noetic/setup.bash` |
| VINS 不输出 odometry | 初始化未完成 | 等待 10-20 秒，需要足够运动激励 |

---

## 16. 许可

- VINS-Fusion: GPLv3 (HKUST Aerial Robotics Group)
- hiking_route_localization: MIT
- 第三方: Ceres (BSD), OpenCV (Apache 2.0), Eigen (MPL2)

---

*文档生成日期: 2026-06-28*
*项目路径: /home/kai/AI/*
