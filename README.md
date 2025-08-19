# Sonar 3D Reconstruction

소나 센서 융합을 통한 3D 해저 지형 재구성 시스템 (Phase 1-3 완료)

## Overview

수상 플랫폼에서 Octree 기반 동적 매핑 시스템을 이용한 실시간 3D 해저 지형 재구성 시스템입니다.

**주요 성과**:
- 메모리 효율성: Dense grid 대비 29~93배 절약  
- 실시간 처리: ~1.5 fps 성능
- 정밀 동기화: ±0.025초 타임스탬프 정확도

## Quick Start

```bash
cd /workspace/ros2_ws/src/sonar_3d_reconstruction/test

# 기본 실행 (권장)
python3 demos/demo_octree_boat_mapping.py --dynamic-map --num-frames 5

# 노이즈 필터링 적용 (최적)
python3 demos/demo_octree_boat_mapping.py --dynamic-map --num-frames 5 --fov-margin 15
```

## System Architecture

### Sensor Configuration
- **Oculus M750D**: 해저 지형 매핑용 멀티빔 소나 (90° 틸트)
- **Livox MID360**: 보트 위치 추정용 LiDAR 
- **Fast-LIO**: 실시간 위치 및 자세 추정

### Coordinate Transformation Pipeline
3단계 변환: **Sonar → Boat → Map**
- 소나 좌표계: +X(전방), +Y(우측), +Z(하방)  
- 맵 좌표계: +X(수평전방), +Y(수평우측), +Z(위)
- Quaternion 기반 정밀 회전 처리

## Core Features (완료)

### Octree Dynamic Mapping
- **동적 맵 확장**: 고정 경계 없음, 보트 이동에 따른 자동 확장
- **Sparse Storage**: 관측된 voxel만 저장하여 메모리 최적화
- **베이지안 확률 업데이트**: Log-odds 누적을 통한 확신도 증가

### Visualization System  
- **확률 기반 색상**: 파랑(자유) → 노랑(미지) → 빨강(점유)
- **적응형 투명도**: 확률 < 0.5는 반투명, ≥ 0.5는 불투명
- **FOV 노이즈 필터링**: 가장자리 데이터 제거로 품질 향상

## Core Components

### ROS2 Nodes

#### `octree_sonar_mapper_node.py`
ROS2 실시간 처리 노드:
- Sonar image 및 odometry 구독
- PointCloud2 3D map 퍼블리싱
- TF2 transform broadcasting (body → sonar_link)
- YAML 기반 파라미터 설정

### Core Libraries

#### `octree_mapper.py`
핵심 매핑 라이브러리:
- `SimpleOctree`: Dictionary 기반 sparse voxel 저장소
- `OctreeBasedSonarMapperWithBoat`: 3단계 좌표 변환 및 매핑
- Full 3-axis rotation support (roll, pitch, yaw)

### `demo_octree_boat_mapping.py` 
통합 데모 애플리케이션:
- 멀티프레임 순차 처리
- 소나-odometry 타임스탬프 동기화  
- 실시간 3D 시각화

### Supporting Files
- `bag_processor.py`: ROS2 bag 파일 처리
- `config.py`: 시스템 설정 관리
- `visualization_utils.py`: 3D 시각화 유틸리티

## Usage Examples

### Basic Usage
```bash
# 기본 매핑 (5프레임)
python3 demos/demo_octree_boat_mapping.py --dynamic-map --num-frames 5

# 고해상도 매핑
python3 demos/demo_octree_boat_mapping.py --dynamic-map --resolution 0.02 --num-frames 10

# 특정 구간 처리
python3 demos/demo_octree_boat_mapping.py --start-frame 500 --num-frames 10 --dynamic-map
```

### Advanced Options
```bash
# FOV 필터링 (양쪽 15도 제거)
python3 demos/demo_octree_boat_mapping.py --dynamic-map --fov-margin 15

# 결과 저장
python3 demos/demo_octree_boat_mapping.py --dynamic-map --save-fig results.png --no-show

# 커스텀 데이터 경로
python3 demos/demo_octree_boat_mapping.py \
    --bag-path /path/to/sonar.bag \
    --odometry-csv /path/to/odometry.csv \
    --dynamic-map
```

## Dataset Information

### 장길리 BlueBoat Dataset
**위치**: `/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar/`

**센서 데이터**:
- Oculus Sonar: 1,280 프레임 (~15Hz)
- Livox LiDAR: 864 프레임 (위치 추정용)
- IMU: 17,279 프레임 (~200Hz)

**시간 동기화**: Unix epoch time vs bag 상대시간 문제를 상대시간 기반 매칭으로 해결

## Performance Metrics

### Processing Performance
- **처리 속도**: ~1.5 fps (실시간 가능)
- **메모리 효율성**: Dense 대비 29~93배 절약
- **타임스탬프 정밀도**: ±0.025초 매칭 정확도

### Memory Comparison
```
벤치마크 (10프레임):
├─ Dense Grid: 2,730,000 voxels
├─ Octree: 94,021 voxels  
└─ 절약율: 29배 효율성
```

## Troubleshooting

### Memory Issues
```bash
# 프레임 수 줄이기
python3 demos/demo_octree_boat_mapping.py --dynamic-map --num-frames 3

# 해상도 낮추기
python3 demos/demo_octree_boat_mapping.py --dynamic-map --resolution 0.05
```

### Visualization Issues  
```bash
# 시각화 없이 실행
python3 demos/demo_octree_boat_mapping.py --dynamic-map --no-show

# 다른 시각화 모드
python3 demos/demo_octree_boat_mapping.py --viz-mode points
```

## Project Status

### Completed (Phase 1-3)
✅ Octree 동적 매핑 시스템  
✅ 3단계 좌표 변환 파이프라인 (3-axis rotation support)  
✅ 베이지안 확률 시각화  
✅ 실시간 처리 성능 (~1.5 fps)  
✅ 메모리 최적화 (29~93배 절약)
✅ ROS2 실시간 노드 시스템
✅ TF2 transform broadcasting
✅ YAML 기반 동적 설정

### Next Phase (Phase 4)
🔲 LiDAR-SLAM 통합 개선
🔲 멀티센서 융합 (Oculus + Ping360 + LiDAR)  
🔲 ROV 탐지 및 추적 알고리즘  
🔲 실시간 센서 융합 노드

## Build & Development

```bash
# ROS2 워크스페이스 빌드
cd /workspace/ros2_ws && colcon build
source /workspace/ros2_ws/install/setup.bash

# ROS2 노드 실행
ros2 launch sonar_3d_reconstruction octree_mapper.launch.py

# 또는 직접 실행
ros2 run sonar_3d_reconstruction octree_sonar_mapper_node --ros-args --params-file /workspace/ros2_ws/install/sonar_3d_reconstruction/share/sonar_3d_reconstruction/config/octree_mapper.yaml

# 데이터 재생 테스트
ros2 bag play /workspace/data/3_janggil_ri/.../sonar-scenario1-v1.deg90.bag/
```