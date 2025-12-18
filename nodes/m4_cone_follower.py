#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class M4ConeFollowerLiDAR:
    def __init__(self):
        rospy.init_node("m4_cone_follower", anonymous=False)
        
        # === 파라미터 설정 ===
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/limo/mode")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        
        # 제어 상수
        self.target_speed = 0.18      
        self.kp_steer = 1.5           
        self.safe_margin = 0.25       # 벽(라바콘) 반발력 거리
        
        self.is_active = False

        # === Publisher & Subscriber ===
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo("[M4] LiDAR Base Ready. Mode switch will be instant.")

    def mode_cb(self, msg):
        if msg.data == "M4_RUN":
            self.is_active = True
            rospy.loginfo("[M4] LiDAR Navigation Started!")
        elif msg.data in ["LANE", "M3_RUN", "IDLE"]:
            self.is_active = False
            self.stop_robot()

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def scan_cb(self, msg):
        if not self.is_active: return

        # 1. 데이터 전처리: 전방 +-60도 필터링
        ranges = np.array(msg.ranges)
        angles = np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        
        # 유효 거리(1.0m) 및 각도 필터링
        mask = (ranges > 0.05) & (ranges < 1.0) & (np.abs(angles) < 60)
        filtered_ranges = ranges[mask]
        filtered_angles = angles[mask]

        # [수정] 라바콘이 감지되지 않으면 즉시 LANE 모드로 복귀
        if len(filtered_ranges) < 3: # 노이즈 고려하여 3개 미만 시 종료
            rospy.logwarn("[M4] No cones detected! Immediate switch to LANE.")
            self.finish_mission()
            return

        # 2. 목표점 계산 및 반발력(Potential Field) 적용
        target_angle = np.mean(filtered_angles)
        repulsive_force = self.calculate_repulsion(filtered_ranges, filtered_angles)

        # 3. 제어 명령 발행
        twist = Twist()
        twist.linear.x = self.target_speed
        twist.angular.z = (np.radians(target_angle) * self.kp_steer) + repulsive_force
        self.pub_cmd.publish(twist)

    def calculate_repulsion(self, ranges, angles):
        """측면 라바콘으로부터의 반발력 계산 (수학적 중앙 유지)"""
        force = 0.0
        for r, a in zip(ranges, angles):
            if r < self.safe_margin:
                # 거리에 반비례하여 핸들을 중앙으로 밀어줌
                force -= np.sign(a) * (self.safe_margin - r) * 2.5
        return force

    def finish_mission(self):
        """미션 종료 및 모드 전환"""
        self.stop_robot()
        self.pub_mode.publish("LANE") # 모드를 LANE으로 변경하여 lane_follower 구동
        self.is_active = False

if __name__ == "__main__":
    M4ConeFollowerLiDAR()
    rospy.spin()