# LIMO 시뮬레이션 코드 v2

LIMO 로봇 시뮬레이션을 위한 ROS 패키지 (v2)입니다.

## 🎯 개발 목표

**로컬 데이터 처리 기반 주행 시스템**
- 경로(Path) 파일 없이 실시간 센서 데이터만 사용
- 맵(Map) 없이 순수 센서 기반 주행
- LiDAR, 카메라 등 로컬 센서 데이터를 활용한 자율 주행

## 📁 패키지 구조

```
sim_code_v2/
├── nodes/              # ROS 노드 스크립트
│   ├── simple_controller.py    # 간단한 컨트롤러 예제
│   └── teleop_keyboard.py      # 키보드 조종 노드
├── launch/             # Launch 파일
│   └── test_simulation.launch  # 시뮬레이션 테스트 launch
├── config/             # 설정 파일 (향후 사용)
├── scripts/            # 유틸리티 스크립트 (향후 사용)
└── rviz/               # RViz 설정 파일 (향후 사용)
```

## 🚀 빠른 시작

### 1. 워크스페이스 빌드

```bash
cd ~/ahns_limo_ws
catkin_make
source devel/setup.bash
```

### 2. 시뮬레이션 실행

#### 방법 1: 단일 launch 파일로 실행 (추천)

```bash
roslaunch sim_code_v2 test_simulation.launch
```

이 명령어는 다음을 실행합니다:
- Gazebo 시뮬레이터 (limo_ackerman.launch)
- LIMO 로봇 모델
- RViz 시각화

#### 방법 2: 터미널을 분리하여 실행

**터미널 1: Gazebo 시뮬레이터**
```bash
cd ~/ahns_limo_ws
source devel/setup.bash
roslaunch limo_gazebo_sim limo_ackerman.launch
```

**터미널 2: 컨트롤러 노드**
```bash
cd ~/ahns_limo_ws
source devel/setup.bash

# 옵션 1: 간단한 자동 컨트롤러
rosrun sim_code_v2 simple_controller.py

# 옵션 2: 키보드 조종
rosrun sim_code_v2 teleop_keyboard.py
```

### 3. 키보드 조종 방법

`teleop_keyboard.py` 노드를 실행한 경우:

```
이동:
   u    i    o
   j    k    l
   m    ,    .

i/k: 전진/후진
j/l: 좌회전/우회전
스페이스 키: 정지
q: 종료
```

## 📝 주요 노드 설명

### simple_controller.py
- 간단한 전진 명령을 발행하는 예제 노드
- `/cmd_vel` 토픽으로 0.5 m/s 속도로 직진
- 개발 시작점으로 활용 가능

### teleop_keyboard.py
- 키보드로 LIMO를 실시간 조종하는 노드
- 테스트 및 디버깅에 유용

## 🔧 개발 가이드

### 새 노드 추가하기

1. `nodes/` 폴더에 Python 스크립트 작성
2. 파일에 실행 권한 부여: `chmod +x nodes/your_node.py`
3. `CMakeLists.txt`의 `catkin_install_python`에 추가:
   ```cmake
   catkin_install_python(PROGRAMS
     nodes/simple_controller.py
     nodes/your_node.py  # 여기에 추가
     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
   )
   ```
4. 워크스페이스 다시 빌드: `catkin_make`

### 로컬 데이터 처리 노드 개발

이 패키지에서는 **경로나 맵 없이** 다음 센서 데이터를 활용하여 주행합니다:

- `/scan` (sensor_msgs/LaserScan): LiDAR 스캔 데이터
- `/usb_cam/image_raw` (sensor_msgs/Image): 카메라 이미지
- `/odom` (nav_msgs/Odometry): 오도메트리 정보
- `/imu` (sensor_msgs/Imu): IMU 데이터

예시: LiDAR 기반 장애물 회피, 카메라 기반 차선 추종 등

## 📊 주요 토픽

- `/cmd_vel` (geometry_msgs/Twist): 속도 명령 (출력)
- `/odom` (nav_msgs/Odometry): 오도메트리 정보 (입력)
- `/scan` (sensor_msgs/LaserScan): LiDAR 스캔 데이터 (입력)
- `/usb_cam/image_raw` (sensor_msgs/Image): 카메라 이미지 (입력)
- `/imu` (sensor_msgs/Imu): IMU 데이터 (입력)

토픽 확인:
```bash
rostopic list
rostopic echo /scan
rostopic echo /odom
```

## 📚 참고 자료

- LIMO 시뮬레이션 패키지: `simulator/limo_gazebo_sim`
- LIMO 설명 패키지: `simulator/limo_description`
- v1 패키지 (전역 경로 기반): `sim_code_v1/`

## 🔄 버전 정보

- **v1**: 전역 경로(Global Path) 기반 네비게이션 시스템
- **v2**: 로컬 데이터 처리 기반 주행 시스템 (현재 버전)
