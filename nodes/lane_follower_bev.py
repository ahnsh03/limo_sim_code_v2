#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import signal
import atexit
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

class LaneFollowerBEVFinal:
    def __init__(self):
        rospy.init_node("lane_follower_bev_final", anonymous=False)

        # === 토픽 설정 ===
        self.img_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw/compressed")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/limo/mode")

        # === [튜닝] 주행 파라미터 ===
        self.base_speed = 0.35      # 기본 주행 속도
        self.steer_k = 2.0          # 조향 민감도
        self.max_steer = 1.0        # 최대 조향각 제한

        # 필터링 (부드러운 주행)
        self.steer_alpha = 0.4      # 반응성 (클수록 민감)
        self.steer_rate = 0.5       # 변화율 제한

        # [색상] 어두운 아스팔트 도로 추출 (흰색 영역으로 변환됨)
        # HSV 명도(V) 기준: 85 이하인 어두운 영역
        self.lower_hsv = np.array([0, 0, 0])
        self.upper_hsv = np.array([180, 255, 85]) 

        # BEV 좌표 (시뮬레이터 최적화)
        self.margin_x = 20          
        self.margin_y = 216         # 상단 45% 지점부터 바닥 인식
        self.dst_margin_x = 150     

        self.steer_f = 0.0
        self.current_steer = 0.0
        
        # [수정 1] 안전을 위해 False로 시작 (명령 대기)
        self.is_active = False       
        
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        # 디버그 이미지 발행 (rqt_image_view에서 확인 가능)
        self.pub_debug = rospy.Publisher("/lane/bev_final/compressed", CompressedImage, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.img_topic, CompressedImage, self.img_cb)

        rospy.loginfo("[LANE_FINAL] Ready. Waiting for 'LANE' command.")

    def mode_cb(self, msg):
        # 모드 신호 처리
        if msg.data == "LANE":
            self.is_active = True
            rospy.loginfo("[LANE] 🟢 Mode set to LANE. Driving started.")
        else:
            self.is_active = False
            self.stop_robot()
            rospy.loginfo(f"[LANE] 🔴 Mode set to {msg.data}. Robot stopped.")

    def stop_robot(self):
        """로봇 정지 명령 발행"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd.publish(twist)
        # rospy.loginfo("[LANE] Robot Stopped.")

    def img_cb(self, msg):
        if not self.is_active: return

        twist = Twist()
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]

            # 1. 색상 필터링 (어두운 도로 = 흰색 마스크)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

            # 2. BEV 변환
            src_pts = np.float32([
                (0, h), (self.margin_x, self.margin_y),
                (w - self.margin_x, self.margin_y), (w, h)
            ])
            dst_pts = np.float32([
                (self.dst_margin_x, h), (self.dst_margin_x, 0), 
                (w - self.dst_margin_x, 0), (w - self.dst_margin_x, h)
            ])
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp_img = cv2.warpPerspective(mask, matrix, (w, h))

            # 3. 가장 큰 도로 영역 찾기 (노이즈 제거)
            contours, _ = cv2.findContours(warp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_cx = w // 2 # 기본값: 화면 중앙
            found_road = False

            # 디버그용 이미지 생성 (컬러)
            debug_img = cv2.cvtColor(warp_img, cv2.COLOR_GRAY2BGR)

            if len(contours) > 0:
                # 면적이 가장 큰 덩어리(메인 도로) 선택
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)

                if M["m00"] > 100: # 최소 면적 확인
                    target_cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    found_road = True
                    
                    # [시각화] 목표 지점(초록색 원)과 외곽선 표시
                    cv2.circle(debug_img, (target_cx, cy), 10, (0, 255, 0), -1)
                    cv2.drawContours(debug_img, [c], -1, (0, 255, 255), 2)

            # 화면 중앙선 표시 (파란색)
            cv2.line(debug_img, (w//2, 0), (w//2, h), (255, 0, 0), 2)

            # 4. 디버그 이미지 발행
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            self.pub_debug.publish(debug_msg)

            # 5. 주행 제어
            if found_road:
                # 에러 계산: 화면중앙 - 도로중심
                error = (w // 2) - target_cx 
                
                # 조향각 계산
                raw_steer = (error * math.pi / w) * self.steer_k

                # 스무딩 필터 적용
                self.steer_f = (1.0 - self.steer_alpha) * self.steer_f + (self.steer_alpha * raw_steer)
                delta = self.steer_f - self.current_steer
                delta = np.clip(delta, -self.steer_rate, self.steer_rate)
                self.current_steer = np.clip(self.current_steer + delta, -self.max_steer, self.max_steer)
                
                # 커브 감속
                speed = self.base_speed
                if abs(self.current_steer) > 0.5: speed *= 0.8

                twist.linear.x = speed
                twist.angular.z = self.current_steer
            else:
                # 도로를 놓쳤을 때 정지
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.logwarn_throttle(1.0, "[BEV] Road Lost!")

            self.pub_cmd.publish(twist)

        except Exception as e:
            rospy.logerr(f"[LANE_FINAL] Error: {e}")
            self.stop_robot()

if __name__ == "__main__":
    # [수정 2] 안전한 종료를 위한 핸들러 등록
    node = None
    try:
        node = LaneFollowerBEVFinal()
        
        # 종료 시 호출될 함수
        def cleanup():
            if node is not None:
                rospy.loginfo("[LANE] Shutting down, stopping robot...")
                node.stop_robot()
        
        # 시그널 핸들러 (Ctrl+C 등)
        def signal_handler(signum, frame):
            cleanup()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass