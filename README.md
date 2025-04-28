# Bolt Fastener

A ROS2-based robotic system for automated bolt detection and fastening using computer vision and robotic control.

## Overview

This project implements an automated bolt fastening system that uses computer vision to detect bolts and a robotic arm to perform the fastening operation. The system utilizes two cameras (head and hand cameras) for visual feedback and precise positioning.

## Features

- **Bolt Detection**: Real-time bolt detection using computer vision
- **3D Pose Estimation**: Accurate 3D position and orientation estimation of detected bolts
- **Robotic Control**: Integration with ARMstrong robot arm for precise motion control
- **Multi-stage Process**:
  - Detection and alignment
  - Docking (positioning)
  - Insertion (fastening)
- **Real-time Visualization**: OpenCV-based visualization of:
  - Camera feeds
  - Detection results
  - Status information
  - Performance metrics

## System Requirements

- ROS2 (tested with [version])
- Python 3.x
- OpenCV
- NumPy
- ARMstrong robot arm
- RGB-D cameras (head and hand cameras)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd bolt_fastener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the ROS2 workspace:
```bash
colcon build
source install/setup.bash
```

## Usage

1. Start the bolt fastener node:
```bash
ros2 run bolt_fastener bolt_fastener_node
```

2. The system will:
   - Initialize cameras and visualizations
   - Begin bolt detection
   - Allow manual target selection via mouse click
   - Automatically perform alignment and fastening

### Control Interface

- **Mouse Controls**:
  - Left click on detected bolts to select target
  - Right click to deselect target

- **Keyboard Controls**:
  - `q`: Quit application
  - `f`: Toggle fix pose mode
  - `r`: Reset planning status
  - `a`: Set offset by IMU
  - `i`: Start insertion process

## System States

The system operates in the following states:
- `IDLE`: Initial state
- `ALIGNING`: Aligning with detected bolt
- `ALIGNED`: Successfully aligned
- `TUNED`: Fine-tuned position
- `DOCKING`: Moving to bolt position
- `DOCKED`: Successfully positioned
- `INSERTING`: Performing bolt insertion
- `INSERTED`: Successfully inserted
- `FAILED`: Error state

## Architecture

The system consists of several key components:

1. **BoltFastenerNode**: Main ROS2 node orchestrating the process
2. **BoltDetector**: Handles bolt detection in camera images
3. **PointCloudProcessor**: Processes 3D point cloud data
4. **InsertStatus**: Manages insertion process states

## Troubleshooting
Common issues and solutions:
1. 카메라가 안켜져요: `use_tcp_tunnel` 사용 해보기 (ex. `ros2 run tcp_tunnel client --ros-args -p client_ip:=[your_pc_ip] -p initial_topic_list_file_name:='~/ros2ws/src/ros2_tcp_tunnel/topic_list.yaml'`)
