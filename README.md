# Bolt Fastener ROS2 Package

A ROS2-based system for automated bolt detection and manipulation using computer vision and robotic control.

## Features
- Real-time bolt detection using YOLO
- 3D pose estimation using depth camera
- Interactive GUI with depth and RGB visualization
- Robot arm control integration
- Bolt tracking and grouping
- Performance monitoring

## Requirements
- ROS2 (tested on Humble)
- Python 3.10
- OpenCV
- NumPy
- SciPy
- Ultralytics (YOLO)
- ARMstrong robot control system

## Installation
1. Clone this repository into your ROS2 workspace:
```zsh
cd ~/ros2ws/src
git clone [repository_url]
```

2. Install dependencies:
```zsh
cd ~/ros2ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the package:
```zsh
colcon build --symlink-install --packages-select bolt_fastener
```

## Usage
1. Source your workspace:
```zsh
source ~/ros2ws/install/setup.zsh
```
2. Launch camera node: on ARMstrong pc,
```zsh
ros2 launch backass cam_rs.launch.py
```

2. Launch the node:
```zsh
ros2 run bolt_fastener bolt_fastener_node
```

## Configuration
The system can be configured through ROS2 parameters:
- `confidence_threshold`: Detection confidence threshold (default: 0.5)
- `use_tcp_tunnel`: Use TCP tunnel for camera communication (default: true)
- `camera_namespace`: Camera topic namespace (default: '/camera/camera')

## GUI Controls
- 'q': Quit
- 'f': Toggle detection update
- Left click: Select bolt target
- Trackbars: Adjust depth visualization and detection parameters

## Troubleshooting
Common issues and solutions:
1. 카메라가 안켜져요: `use_tcp_tunnel` 사용 해보기 (ex. `ros2 run tcp_tunnel client --ros-args -p client_ip:=[your_pc_ip] -p initial_topic_list_file_name:='~/ros2ws/src/ros2_tcp_tunnel/topic_list.yaml'`)
 
## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 