#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class M3GateNav:
    def __init__(self):
        rospy.init_node("m3_gate_nav", anonymous=False)

        # === 설정 ===
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/limo/mode")

        # [1] 제어 파라미터
        self.check_dist = 1.5         
        self.cluster_tol = 0.30       
        self.kp_centering = 1.8       
        self.force_pull_gain = 2.5    
        self.force_push_gain = 1.5    
        
        # [2] 가변 세이프 마진
        self.safe_margin = 0.6        
        
        # [3] 주행 제어 파라미터
        self.base_speed = 0.22
        self.delivery_speed = 0.15    
        self.steer_limit = 1.5        
        self.last_steer = 0.0
        self.max_steer_change = 2.3  # 기존 코드의 높은 반응성 유지
        
        # [4] 상태 전환 및 종료 임계값
        self.entry_dist_thresh = 0.40   
        self.exit_dist_thresh = 0.45    
        self.open_space_thresh = 1.2  # 최종 종료 기준

        self.state = "IDLE"
        self.pass_timer = None
        self.entry_stable_start = None

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo(f"[M3] Entry Logic: Original (1.0m Scan) / Exit Logic: Advanced (Both > 1.2m)")

    def mode_cb(self, msg):
        if msg.data == "LANE":
            self.state = "MONITORING"
        elif msg.data == "M3_RUN":
            if self.state not in ["PASS_INSIDE", "PASS_EXIT", "DELIVERY_M4"]:
                self.state = "CENTERING"
        elif msg.data == "M4_RUN":
            self.state = "IDLE"
            self.stop_robot()

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def scan_cb(self, msg):
        if self.state == "IDLE": return

        ranges = np.array(msg.ranges)
        angles = np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        valid = (ranges > 0.05) & (ranges < 4.0) & np.isfinite(ranges)
        r = ranges[valid]
        a = angles[valid]

        if len(r) == 0: return

        # ---------------------------------------------------------
        # [측면 스캔 데이터 구성]
        # ---------------------------------------------------------
        # 기본 스캔 (기존 코드와 동일하게 1.0m 범위 사용)
        l_mask = (a > 80) & (a < 100) & (r < 1.0)
        r_mask = (a > -100) & (a < -80) & (r < 1.0)
        l_side = np.min(r[l_mask]) if np.any(l_mask) else 0.6
        r_side = np.min(r[r_mask]) if np.any(r_mask) else 0.6
        center_error = l_side - r_side

        # 배달 종료 판단용 광역 스캔 (1.5m까지 감시)
        l_side_wide = np.min(r[(a > 80) & (a < 100) & (r < 1.5)]) if np.any((a > 80) & (a < 100) & (r < 1.5)) else 2.0
        r_side_wide = np.min(r[(a > -100) & (a < -80) & (r < 1.5)]) if np.any((a > -100) & (a < -80) & (r < 1.5)) else 2.0

        # =========================================
        # 1. 감시 (MONITORING)
        # =========================================
        if self.state == "MONITORING":
            self.safe_margin = 0.6
            if self.check_trigger(r, a):
                self.pub_mode.publish("M3_RUN")
                self.state = "CENTERING"
                rospy.logwarn("[State] MONITORING -> CENTERING")

        # =========================================
        # 2. 접근 (CENTERING) - 기존 코드 로직 유지
        # =========================================
        elif self.state == "CENTERING":
            if l_side < self.entry_dist_thresh and r_side < self.entry_dist_thresh:
                if self.entry_stable_start is None:
                    self.entry_stable_start = rospy.Time.now()
                elif (rospy.Time.now() - self.entry_stable_start).to_sec() > 0.5:
                    self.state = "PASS_INSIDE"
                    rospy.logwarn("[State] CENTERING -> PASS_INSIDE")
                    return
            
            gate_found, _, gate_ang = self.find_gate_target(r, a)
            if gate_found:
                self.safe_margin = 0.3
                target_steer = np.radians(gate_ang) * self.force_pull_gain + self.calculate_potential_push(r, a)
            else:
                self.safe_margin = 0.6
                target_steer = center_error * self.kp_centering + self.calculate_potential_push(r, a)
            
            rospy.loginfo_throttle(0.5, f"[CENTERING] L:{l_side:.2f} R:{r_side:.2f} | Error: {center_error:.3f}")
            self.drive_smooth(r, a, target_steer, self.base_speed)

        # =========================================
        # 3. 통로 내부 (PASS_INSIDE)
        # =========================================
        elif self.state == "PASS_INSIDE":
            self.safe_margin = 0.15
            if l_side > self.exit_dist_thresh or r_side > self.exit_dist_thresh:
                self.state = "PASS_EXIT"
                self.pass_timer = rospy.Time.now()
                rospy.logwarn("[State] PASS_INSIDE -> PASS_EXIT")
                return

            target_steer = (center_error * self.kp_centering) + self.calculate_potential_push(r, a)
            self.drive_smooth(r, a, target_steer, self.base_speed)

        # =========================================
        # 4. 통로 탈출 후 0.5초 직진 (PASS_EXIT)
        # =========================================
        elif self.state == "PASS_EXIT":
            elapsed = (rospy.Time.now() - self.pass_timer).to_sec()
            if elapsed < 0.5:
                self.drive_raw(0.15, 0.0) 
            else:
                self.state = "DELIVERY_M4"
                rospy.logwarn("[State] DELIVERY_M4 Start")

        # =========================================
        # 5. M4 배달 구역 주행 (DELIVERY_M4) - 양쪽 1.2m 종료 로직 결합
        # =========================================
        elif self.state == "DELIVERY_M4":
            if l_side_wide > self.open_space_thresh and r_side_wide > self.open_space_thresh:
                rospy.logwarn(f"[DELIVERY] BOTH sides > 1.2m -> FINISH")
                self.finish()
                return

            self.safe_margin = 0.6
            # 중앙 유지는 광역 데이터를 기반으로 수행
            error_wide = l_side_wide - r_side_wide
            target_steer = (error_wide * self.kp_centering) + self.calculate_potential_push(r, a)
            
            rospy.loginfo_throttle(0.5, f"[DELIVERY] L:{l_side_wide:.2f} R:{r_side_wide:.2f}")
            self.drive_smooth(r, a, target_steer, self.delivery_speed)

    # === 제어 및 유틸리티 ===
    def drive_smooth(self, r, a, target_w, speed_limit):
        mask_f = (a > -15) & (a < 15) & (r < 0.6)
        speed = 0.12 if np.any(mask_f) else speed_limit
        if speed > 0 and np.any(r[(a > -15) & (a < 15)] < 0.12): speed = -0.1

        target_w = np.clip(target_w, -self.steer_limit, self.steer_limit)
        delta = target_w - self.last_steer
        clamped_delta = np.clip(delta, -self.max_steer_change, self.max_steer_change)
        smooth_w = self.last_steer + clamped_delta
        self.drive_raw(speed, smooth_w)

    def drive_raw(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)
        self.last_steer = w

    def calculate_potential_push(self, r, a):
        push_steer = 0.0
        mask_obs = (r < self.safe_margin) & (a > -90) & (a < 90)
        if np.any(mask_obs):
            obs_r = r[mask_obs]
            obs_a = a[mask_obs]
            forces = (1.0 / (obs_r ** 2)) 
            push_effects = -np.sign(obs_a) * forces
            push_steer = np.sum(push_effects) 
            push_steer = np.clip(push_steer, -2.5, 2.5) * self.force_push_gain
        return push_steer

    def check_trigger(self, r, a):
        mask = (r < self.check_dist) & (a > -90) & (a < 90)
        r_m = r[mask]; a_m = a[mask]
        if len(r_m) < 5: return False 
        rads = np.radians(a_m)
        points = np.stack((r_m * np.cos(rads), r_m * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > self.cluster_tol)[0] + 1
        clusters = np.split(points, split_idx)
        tips = [c[np.argmin(np.linalg.norm(c, axis=1))] for c in clusters if len(c) >= 3]
        if len(tips) >= 2:
            for i in range(len(tips)):
                for j in range(i+1, len(tips)):
                    if 0.9 <= np.linalg.norm(tips[i] - tips[j]) <= 1.5: return True
        return False

    def find_gate_target(self, r, a):
        mask = (r < 1.3) & (a > -50) & (a < 50)
        r_g = r[mask]; a_g = a[mask]
        if len(r_g) < 5: return False, 0, 0
        rads = np.radians(a_g)
        points = np.stack((r_g * np.cos(rads), r_g * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > 0.3)[0] + 1
        clusters = np.split(np.arange(len(r_g)), split_idx)
        valid_clusters = [c for c in clusters if len(c) >= 3]
        for i in range(len(valid_clusters) - 1):
            p1 = points[valid_clusters[i]][np.argmin(np.linalg.norm(points[valid_clusters[i]], axis=1))]
            p2 = points[valid_clusters[i+1]][np.argmin(np.linalg.norm(points[valid_clusters[i+1]], axis=1))]
            if 0.3 <= np.linalg.norm(p1 - p2) <= 0.7:
                mid = (p1 + p2) / 2.0
                return True, np.linalg.norm(mid), np.degrees(np.arctan2(mid[1], mid[0]))
        return False, 0, 0

    def finish(self):
        rospy.logwarn("[State] FINISHED -> M4_RUN")
        self.pub_mode.publish("M4_RUN")
        self.state = "IDLE"
        self.stop_robot()

if __name__ == "__main__":
    try:
        M3GateNav()
        rospy.spin()
    except rospy.ROSInterruptException: pass