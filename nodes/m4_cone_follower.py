#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import signal
import atexit
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

class M4ConeFollower:
    def __init__(self):
        rospy.init_node("m4_cone_follower", anonymous=False)
        
        # === 파라미터 및 토픽 설정 ===
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/limo/mode")
        self.img_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw/compressed")
        
        self.is_active = False
        self.bridge = CvBridge()

        # === [팀원 코드 이식] 제어 변수 ===
        self.steer = 0.0              # 직전 조향값 저장용
        self.one_side_start_time = None # 한쪽만 보일 때 타이머
        self.wait_time_limit = 1.0    # 한쪽만 보일 때 대기 시간 (1초)
        self.target_speed = 0.21      # 주행 속도
        self.offset_x = 150           # 한쪽만 보일 때 목표 지점 보정값

        # === Publisher & Subscriber ===
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.img_topic, CompressedImage, self.image_cb)

        rospy.loginfo("[M4] Ready. Waiting for 'M4_RUN' command...")

    def mode_cb(self, msg):
        """ 모드 콜백: M4_RUN일 때만 동작 """
        if msg.data == "M4_RUN":
            self.is_active = True
            self.one_side_start_time = None
            self.steer = 0.0
            rospy.loginfo("[M4] Started! Searching for RED cones.")
        elif msg.data in ["LANE", "M3_RUN", "IDLE"]:
            self.is_active = False
            self.stop_robot()
            self.one_side_start_time = None

    def stop_robot(self):
        twist = Twist()
        self.pub_cmd.publish(twist)

    def image_cb(self, msg):
        if not self.is_active: return

        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.process_cone_following(frame)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_cone_following(self, frame):
        twist = Twist()
        h, w, _ = frame.shape
        screen_center = w // 2

        # 1. ROI 설정 (하단 70% 영역 사용)
        roi_h = int(h * 0.3)
        roi = frame[roi_h:h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 2. 빨간색 콘 필터링 (팀원 코드의 HSV 값 적용)
        lower_r1 = np.array([0, 120, 80]);  upper_r1 = np.array([10, 255, 255])
        lower_r2 = np.array([170, 120, 80]); upper_r2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_r1, upper_r1)
        mask2 = cv2.inRange(hsv, lower_r2, upper_r2)
        mask_r = cv2.bitwise_or(mask1, mask2)

        # 3. 컨투어 검출 및 무게중심 계산
        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_cones = []  # (cx, cy)
        right_cones = [] # (cx, cy)

        for cnt in contours:
            if cv2.contourArea(cnt) < 200: continue # 노이즈 제거
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 화면 중앙 기준으로 좌우 분류
            if cx < screen_center:
                left_cones.append((cx, cy))
            else:
                right_cones.append((cx, cy))

        # 4. 주행 로직
        if len(left_cones) > 0 or len(right_cones) > 0:
            
            # [CASE 1] 양쪽 콘이 모두 보임 -> 중앙 주행
            if len(left_cones) > 0 and len(right_cones) > 0:
                self.one_side_start_time = None # 타이머 초기화
                
                # y좌표 기준 정렬 (팀원 코드 로직 따름)
                # 가까운 콘(화면 아래쪽)을 보려면 reverse=True가 맞으나, 
                # 팀원 코드는 sorted(key=lambda p: p[1]) 사용 (화면 위쪽 콘 기준일 수 있음)
                # 여기서는 가장 일반적인 '가까운 콘' 기준(y가 큰 값)으로 작성함
                l_target = sorted(left_cones, key=lambda p: p[1], reverse=True)[0]
                r_target = sorted(right_cones, key=lambda p: p[1], reverse=True)[0]
                
                target_x = (l_target[0] + r_target[0]) // 2
                
                # 조향 계산
                error = target_x - screen_center
                twist.linear.x = self.target_speed
                twist.angular.z = - (error / 180.0)
                
                # 조향값 저장 (한쪽만 보일 때 사용하기 위함)
                self.steer = twist.angular.z

            # [CASE 2] 한쪽 콘만 보임 -> 예외 처리
            else:
                if self.one_side_start_time is None:
                    self.one_side_start_time = rospy.Time.now().to_sec()
                
                elapsed = rospy.Time.now().to_sec() - self.one_side_start_time

                # [상태 A] 1초 미만: 직전 조향 유지 (급격한 회전 방지)
                if elapsed < self.wait_time_limit:
                    twist.linear.x = self.target_speed
                    twist.angular.z = self.steer # 직전 값 유지
                
                # [상태 B] 1초 경과: 오프셋(Offset) 적용하여 탈출
                else:
                    target_x = screen_center # 기본값
                    
                    if len(left_cones) > 0:
                        # 왼쪽만 보임 -> 왼쪽 콘보다 오른쪽으로(cx + 150) 이동
                        l_target = sorted(left_cones, key=lambda p: p[0], reverse=True)[0]
                        target_x = l_target[0] + self.offset_x
                    
                    elif len(right_cones) > 0:
                        # 오른쪽만 보임 -> 오른쪽 콘보다 왼쪽으로(cx - 150) 이동
                        r_target = sorted(right_cones, key=lambda p: p[0])[0]
                        target_x = r_target[0] - self.offset_x
                    
                    error = target_x - screen_center
                    twist.linear.x = self.target_speed
                    twist.angular.z = - (error / 180.0)
                    self.steer = twist.angular.z

            self.pub_cmd.publish(twist)

        else:
            # 콘이 아예 안 보임 -> 정지
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub_cmd.publish(twist)
            self.one_side_start_time = None
            
            # (옵션) 미션 완료 로직이 필요하면 여기에 추가
            # self.finish_mission()

    def finish_mission(self):
        self.pub_cmd.publish(Twist())
        rospy.loginfo("[M4] Mission Complete -> LANE")
        self.pub_mode.publish("LANE")
        self.is_active = False

if __name__ == "__main__":
    node = None
    try:
        node = M4ConeFollower()
        
        # 종료 시 자동 멈춤 핸들러 등록
        def cleanup():
            if node is not None:
                rospy.loginfo("[M4] Shutting down, stopping robot...")
                node.stop_robot()
        
        def signal_handler(signum, frame):
            cleanup()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup)
        
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[M4] Interrupted by user")
        if node is not None:
            node.stop_robot()
    except Exception as e:
        rospy.logerr(f"[M4] Error: {e}")
        if node is not None:
            node.stop_robot()
    finally:
        if node is not None:
            node.stop_robot()