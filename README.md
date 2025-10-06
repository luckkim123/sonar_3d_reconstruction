# Sonar 3D Reconstruction

실시간 확률적 3D 해저 지형 재구성 시스템 (ROS2 Humble)

## Overview

Oculus M750D 멀티빔 소나와 Livox MID360 LiDAR를 활용한 실시간 3D 해저 지형 매핑 시스템입니다. feature_extraction_3d 알고리즘 기반으로 완전히 재구현되어 향상된 확률적 매핑과 메모리 효율성을 제공합니다.

**주요 특징**:
- **Adaptive Bayesian Update**: 자유 공간 보호 및 노이즈 감소
- **Sparse Octree Storage**: 메모리 효율적인 동적 맵 확장
- **TF Integration**: Fast-LIO와 완전 통합된 좌표 변환
- **Real-time Configuration**: Build 없이 YAML 수정 즉시 적용

## Quick Start

```bash
# 1. 워크스페이스 빌드
cd /workspace/ros2_ws
colcon build --packages-select sonar_3d_reconstruction
source install/setup.bash

# 2. 전체 시스템 실행 (Fast-LIO + 3D Mapper + RViz + Bag)
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py

# 3. 파라미터 실시간 변경 (build 불필요)
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py \
  sonar_orientation.yaw:=-90.0 voxel_resolution:=0.1
```

## System Architecture

### Core Components

#### `3d_mapper.py`
Feature extraction 기반 확률적 매핑 라이브러리:
- **SimpleOctree**: Sparse voxel storage with adaptive update
- **SonarTo3DMapper**: 3단계 좌표 변환 (Sonar → Base → Map)
- **Adaptive Protection**: Linear interpolation 기반 자유 공간 보호
- **Vertical Aperture**: 20° 빔 확산 정밀 처리

#### `3d_mapper_node.py`
ROS2 실시간 처리 노드:
- **Time Synchronization**: ApproximateTimeSynchronizer (±0.1s)
- **TF Integration**: Fast-LIO의 camera_init/body 프레임 통합
- **Dynamic Publishing**: PointCloud2 및 MarkerArray 지원
- **Parameter Priority**: CLI > YAML > Launch > Node > Library

### Configuration

#### `config/3d_mapper.yaml`
```yaml
sonar_3d_mapper:
  ros__parameters:
    # 소나 센서 파라미터
    horizontal_fov: 130.0        # 수평 FOV (degrees)
    vertical_aperture: 20.0      # 수직 빔 확산 (degrees)
    voxel_resolution: 0.05       # 복셀 크기 (meters)
    
    # 좌표 변환 (degrees)
    sonar_orientation:
      roll: 0.0
      pitch: 90.0   # 하방 (90도)
      yaw: 0.0
    
    # 확률 업데이트
    log_odds_occupied: 1.5
    log_odds_free: -2.0
    adaptive_update: true        # 자유 공간 보호 활성화
```

#### `launch/3d_mapping.launch.py`
- 소스 폴더 직접 참조 (build 없이 설정 변경 가능)
- Fast-LIO 자동 실행 옵션
- Bag 파일 자동 재생 옵션
- RViz 통합 실행

## Parameter System

### Priority Order (Highest → Lowest)
1. **Command Line**: `--ros-args -p param:=value`
2. **YAML File**: `/workspace/ros2_ws/src/sonar_3d_reconstruction/config/3d_mapper.yaml`
3. **Launch File**: Launch 파일 내 파라미터
4. **Node Defaults**: `3d_mapper_node.py` 기본값
5. **Library Defaults**: `3d_mapper.py` 기본값

### Real-time Configuration
YAML 파일을 직접 참조하도록 설정되어 있어, **colcon build 없이** 설정 변경 가능:
```bash
# YAML 파일 수정 후 바로 실행
vim config/3d_mapper.yaml  # 설정 변경
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py  # 즉시 적용
```

## Coordinate System & TF

### TF Tree Integration
```
camera_init (Fast-LIO map frame)
    └── body (robot frame)
        └── sonar_link (from bag tf_static)
```

### Frame Configuration
- **Map Frame**: `camera_init` (Fast-LIO의 맵 좌표계)
- **Robot Frame**: `body` (Fast-LIO의 로봇 좌표계)
- **Sonar Frame**: `sonar_link` (bag 파일의 tf_static 사용)

### Coordinate Conventions
```yaml
# Oculus M750d (실제 센서)
Sonar: +X forward, +Y right, +Z down

# Map (Fast-LIO)
Map: +X forward, +Y left, +Z up
```

## Usage Examples

### Basic Launch
```bash
# 전체 시스템 (권장)
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py

# Fast-LIO 없이 (odometry 이미 있는 경우)
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py launch_fast_lio:=false

# RViz 없이
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py launch_rviz:=false

# Bag 파일 자동 재생
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py play_bag:=true
```

### Parameter Override
```bash
# Launch 시 파라미터 변경
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py \
  sonar_orientation.yaw:=-90.0 \
  sonar_orientation.pitch:=45.0 \
  voxel_resolution:=0.1

# 노드 직접 실행 시
ros2 run sonar_3d_reconstruction 3d_mapper_node.py --ros-args \
  -p horizontal_fov:=70.0 \
  -p voxel_resolution:=0.05
```

### KIRO Water Tank Dataset
```bash
# 기본 bag 파일 경로 설정됨
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py

# 다른 bag 파일 사용
ros2 launch sonar_3d_reconstruction 3d_mapping.launch.py \
  bag_file:=/workspace/data/2_kiro_watertank/20250926_blueboat_sonar_lidar/oculus-tilt60-gain50-freq2
```

## Key Features

### Adaptive Bayesian Update
- **Linear Interpolation**: 확률에 따른 가중치 조정
- **Free Space Protection**: 높은 확신도의 자유 공간 보호
- **Noise Reduction**: 적응형 업데이트로 노이즈 감소

### Memory Efficiency
- **Sparse Storage**: 관측된 복셀만 저장
- **Dynamic Expansion**: 고정 경계 없이 자동 확장
- **Octree Structure**: 계층적 공간 분할

### Visualization
- **PointCloud2**: 확률값을 intensity로 표시
- **MarkerArray**: 색상 기반 확률 시각화 (옵션)
- **RViz Integration**: TF, Path, Map 통합 표시

## Build & Development

### Dependencies
- ROS2 Humble
- PCL (Point Cloud Library)
- Eigen3
- OpenCV (cv_bridge)
- Python 3.10+

### Build Instructions
```bash
# 의존성 설치
sudo apt update
sudo apt install ros-humble-pcl-* ros-humble-cv-bridge

# 빌드
cd /workspace/ros2_ws
colcon build --packages-select sonar_3d_reconstruction

# 환경 설정
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

### Development Mode
```bash
# 소스에서 직접 실행 (개발 시)
cd /workspace/ros2_ws
source install/setup.bash
ros2 run sonar_3d_reconstruction 3d_mapper_node.py --ros-args \
  --params-file src/sonar_3d_reconstruction/config/3d_mapper.yaml
```

## Troubleshooting

### Common Issues

#### TF Conflicts
```bash
# bag 파일의 tf_static과 충돌 시
# config/3d_mapper.yaml에서 publish_tf: false 설정
```

#### Memory Issues
```bash
# 해상도 낮추기
voxel_resolution: 0.1  # 0.05 → 0.1

# 처리 범위 제한
max_range: 5.0  # 10.0 → 5.0
```

#### Time Synchronization
```bash
# 동기화 허용 오차 조정 (현재 0.1초)
# 필요시 코드에서 slop 파라미터 수정
```

## Performance Metrics

- **Processing Rate**: ~1.5 fps (실시간 가능)
- **Memory Usage**: Sparse octree로 29-93x 절약
- **Synchronization**: ±0.1s 타임스탬프 매칭
- **Voxel Resolution**: 0.05m (설정 가능)

## Recent Updates (2025-09-29)

### Complete Refactoring
- ✅ feature_extraction_3d 기반 재구현
- ✅ Adaptive update 메커니즘 추가
- ✅ Fast-LIO TF tree 완전 통합
- ✅ 소스 YAML 직접 참조 (빌드 불필요)
- ✅ 파라미터 우선순위 시스템 구현
- ✅ 향상된 로깅 및 시각화

### Breaking Changes
- ❌ octree_mapper_node.py → 3d_mapper_node.py
- ❌ mapping.launch.py → 3d_mapping.launch.py
- ❌ test/ 폴더 제거 (통합 완료)

## License & Contact

ROS2 패키지 표준 라이선스를 따릅니다.

Repository: https://github.com/luckkim123/sonar_3d_reconstruction.git