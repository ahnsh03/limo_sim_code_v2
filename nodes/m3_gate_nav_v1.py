#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
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

        # [1] 인식 파라미터 (단순하고 강건하게)
        self.check_dist = 1.8         
        self.cluster_tol = 0.30       # 30cm 이내면 한 덩어리 ('ㄱ'자 허용)
        self.min_wall_length = 0.20   
        
        # [2] 제어 블랜딩 파라미터 (Blending)
        # 거리에 따라 Gap제어와 Align제어 비중을 조절
        self.blend_min_dist = 0.6     # 이보다 가까우면 100% Align 제어
        self.blend_max_dist = 1.4     # 이보다 멀면 100% Gap 제어
        
        # Gap Aiming 게인 (멀 때 주로 작동)
        self.kp_gap = 1.8             
        
        # Wall Alignment 게인 (가까울 때 주로 작동)
        self.kp_align = 2.5           # 기울기
        self.kp_dist = 1.5            # 중앙 유지
        
        self.base_speed = 0.22
        
        # 좁은 게이트(출구) 파라미터
        self.narrow_min = 0.35
        self.narrow_max = 0.65 

        self.state = "IDLE"          
        self.pass_start_time = None
        self.last_target_angle = 0.0

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo("[M3] Ready. Algorithm: Sector Slicing + Control Blending")

    def mode_cb(self, msg):
        if msg.data == "LANE":
            self.state = "MONITORING"
            rospy.loginfo("[M3] Monitoring...")
        elif msg.data == "M3_RUN":
            if self.state != "PASS_GATE":
                self.state = "CENTERING"
                rospy.loginfo("[M3] Switch to CENTERING (Blending Active)")
        elif msg.data == "M4_RUN":
            self.state = "IDLE"
            self.stop_robot()

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def scan_cb(self, msg):
        if self.state == "IDLE": return

        ranges = np.array(msg.ranges)
        angles = np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        valid = (ranges > 0.1) & (ranges < 4.0) & np.isfinite(ranges)
        r = ranges[valid]
        a = angles[valid]

        if len(r) == 0: return

        # =========================================
        # 1. 감시 (MONITORING)
        # =========================================
        if self.state == "MONITORING":
            # 1.8m 이내 데이터만 사용
            mask = (r < self.check_dist) & (a > -90) & (a < 90)
            r_m = r[mask]
            a_m = a[mask]
            
            if len(r_m) < 10: return

            rads = np.radians(a_m)
            xs = r_m * np.cos(rads)
            ys = r_m * np.sin(rads)
            points = np.stack((xs, ys), axis=1)

            # 단순 거리 클러스터링 ('ㄱ'자도 하나로 묶임)
            diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
            split_idx = np.where(diffs > self.cluster_tol)[0] + 1
            clusters = np.split(points, split_idx)
            
            # 유효한 장애물 후보 추출 (직선 피팅)
            lines = []
            for clust in clusters:
                if len(clust) < 5: continue
                try:
                    m, c = np.polyfit(clust[:, 0], clust[:, 1], 1)
                    mse = np.mean((clust[:, 1] - (m * clust[:, 0] + c))**2)
                    # MSE 조건을 완화하여 'ㄱ'자도 일단 후보로 잡음
                    if mse < 0.1: 
                         lines.append({'m': m, 'c': c})
                except: continue

            # 두 장애물 사이 거리 확인
            found = False
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    l1, l2 = lines[i], lines[j]
                    avg_m = (l1['m'] + l2['m']) / 2.0
                    dist = abs(l1['c'] - l2['c']) / math.sqrt(avg_m**2 + 1)
                    
                    # 평행 조건 완화, 거리 조건 확인
                    if 0.9 <= dist <= 1.5:
                        found = True
                        rospy.logwarn(f"[M3] Trigger! Gap Dist: {dist:.2f}m")
                        break
                if found: break

            if found:
                self.pub_mode.publish("M3_RUN")
                self.state = "CENTERING"

        # =========================================
        # 2. 정렬 (CENTERING) - 블랜딩 제어 적용
        # =========================================
        elif self.state == "CENTERING":
            if self.find_narrow_gate(r, a): return

            # [1] 데이터 전처리
            # 왼쪽/오른쪽 전체 영역 (가까운 점 찾기용, 'ㄱ'자 전체 포함)
            mask_l_all = (a > 0) & (a < 110) & (r < 2.0)
            mask_r_all = (a > -110) & (a < 0) & (r < 2.0)
            
            # [핵심] 기울기 계산용 측면 섹터 ('ㄱ'자의 앞면 제거)
            # 30도 이상의 측면 데이터만 사용하여 polyfit 오차를 줄임
            mask_l_side = (a > 30) & (a < 110) & (r < 2.0)
            mask_r_side = (a > -110) & (a < -30) & (r < 2.0)

            # --- A. Gap Aiming (위치 제어) ---
            l_min_dist = r[mask_l_all].min() if np.any(mask_l_all) else 2.0
            r_min_dist = r[mask_r_all].min() if np.any(mask_r_all) else 2.0
            
            # 가장 가까운 점(코너)의 각도 찾기
            l_min_idx = np.argmin(r[mask_l_all]) if np.any(mask_l_all) else 0
            r_min_idx = np.argmin(r[mask_r_all]) if np.any(mask_r_all) else 0
            l_angle_raw = a[mask_l_all][l_min_idx] if np.any(mask_l_all) else 45
            r_angle_raw = a[mask_r_all][r_min_idx] if np.any(mask_r_all) else -45

            # 갭의 중심 각도
            gap_center_angle = (l_angle_raw + r_angle_raw) / 2.0
            steer_gap = np.radians(gap_center_angle) * self.kp_gap

            # --- B. Wall Alignment (자세 제어) ---
            # 측면 데이터만 사용하여 기울기 계산 ('ㄱ'자 앞면 배제)
            slope_angle = 0.0
            dist_error = 0.0
            
            has_slope = False
            
            # 왼쪽 벽 기울기
            if np.sum(mask_l_side) > 5:
                lx = r[mask_l_side] * np.cos(np.radians(a[mask_l_side]))
                ly = r[mask_l_side] * np.sin(np.radians(a[mask_l_side]))
                m_l, c_l = np.polyfit(lx, ly, 1)
                slope_angle += np.degrees(np.arctan(m_l)) # 왼쪽은 +기울기면 우측조향 필요
                dist_error += (abs(c_l) - 0.7) # 목표거리 0.7m (1.4m폭 가정)
                has_slope = True
            
            # 오른쪽 벽 기울기
            if np.sum(mask_r_side) > 5:
                rx = r[mask_r_side] * np.cos(np.radians(a[mask_r_side]))
                ry = r[mask_r_side] * np.sin(np.radians(a[mask_r_side]))
                m_r, c_r = np.polyfit(rx, ry, 1)
                slope_angle += np.degrees(np.arctan(m_r)) 
                dist_error -= (abs(c_r) - 0.7) # 오른쪽은 - 빼줌
                has_slope = True

            # 평균 기울기와 거리 오차로 조향값 계산
            # 양쪽 다 보이면 평균, 하나만 보이면 그 값 사용
            if np.sum(mask_l_side) > 5 and np.sum(mask_r_side) > 5:
                # 둘 다 보일 때는 거리 에러를 (L - R) 차이로 계산하는 게 더 정확함
                dist_error = (abs(c_l) - abs(c_r)) 
            
            steer_align = (np.radians(slope_angle) * self.kp_align) + (dist_error * self.kp_dist)

            # --- C. Control Blending (거리 기반 혼합) ---
            # 벽과의 최단 거리 계산
            curr_dist = min(l_min_dist, r_min_dist)
            
            # 알파 값 계산 (0.0 ~ 1.0)
            # 멀면(1.4m) alpha=1.0 (Gap 위주), 가까우면(0.6m) alpha=0.0 (Align 위주)
            alpha = (curr_dist - self.blend_min_dist) / (self.blend_max_dist - self.blend_min_dist)
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # 최종 조향 (가중치 적용)
            final_steer = (alpha * steer_gap) + ((1.0 - alpha) * steer_align)
            
            # 만약 측면 벽(기울기) 감지가 안 됐다면 무조건 Gap Aiming 사용
            if not has_slope:
                final_steer = steer_gap
            
            final_steer = np.clip(final_steer, -1.5, 1.5)

            # --- D. 속도 제어 ---
            # 전방 1.5m, 폭 0.8m 박스 내 장애물 확인
            mask_f = (a > -20) & (a < 20) & (r < 1.5)
            min_front = np.min(r[mask_f]) if np.any(mask_f) else 2.0
            
            if min_front < 0.4: speed = -0.1
            elif min_front < 0.9: speed = 0.12 # 천천히 진입
            else: speed = self.base_speed      # 빠르게 접근

            self.drive_raw(speed, final_steer)
            
            # 디버깅
            # rospy.loginfo_throttle(0.5, f"Dist:{curr_dist:.2f} Alpha:{alpha:.2f} Steer:{final_steer:.2f}")

        # =========================================
        # 3. 통과 (PASS_GATE)
        # =========================================
        elif self.state == "PASS_GATE":
            elapsed = (rospy.Time.now() - self.pass_start_time).to_sec()
            twist = Twist()
            if elapsed < 2.5:
                twist.linear.x = 0.2
                twist.angular.z = np.radians(self.last_target_angle) * 0.3
                self.pub_cmd.publish(twist)
            else:
                self.finish()

    def find_narrow_gate(self, r, a):
        mask = (r < 1.6) & (a > -45) & (a < 45)
        r_g = r[mask]
        a_g = a[mask]
        if len(r_g) < 5: return False

        rads = np.radians(a_g)
        xs = r_g * np.cos(rads)
        ys = r_g * np.sin(rads)
        points = np.stack((xs, ys), axis=1)

        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > 0.3)[0] + 1
        clusters = np.split(np.arange(len(r_g)), split_idx)
        valid_clusters = [c for c in clusters if len(c) >= 3]

        for i in range(len(valid_clusters) - 1):
            c_r = valid_clusters[i]; c_l = valid_clusters[i+1]
            p1 = points[c_r[-1]]; p2 = points[c_l[0]]
            width = np.linalg.norm(p1 - p2)
            if self.narrow_min <= width <= self.narrow_max:
                if abs(p1[0] - p2[0]) < 0.3:
                    mid_point = (p1 + p2) / 2.0
                    target_ang = np.degrees(np.arctan2(mid_point[1], mid_point[0]))
                    rospy.logwarn(f"[M3] EXIT FOUND! W:{width:.2f} Ang:{target_ang:.1f}")
                    self.last_target_angle = target_ang
                    self.pass_start_time = rospy.Time.now()
                    self.state = "PASS_GATE"
                    return True
        return False

    def drive_raw(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

    def finish(self):
        rospy.logwarn("[M3] Complete -> M4")
        self.pub_mode.publish("M4_RUN")
        self.state = "IDLE"
        self.stop_robot()

if __name__ == "__main__":
    try:
        M3GateNav()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass