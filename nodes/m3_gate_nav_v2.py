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

        # [1] 포텐셜 필드 파라미터 (Push & Pull)
        self.force_pull_gain = 1.8    # 당기는 힘 (게이트 방향)
        self.force_push_gain = 0.8    # 미는 힘 (장애물 회피)
        self.safe_margin = 0.6        # 이 거리 안쪽 장애물만 반발력 작용
        
        # [2] 주행 파라미터
        self.base_speed = 0.22        # 평상시 속도
        self.min_speed = 0.08         # 장애물 근접 시 최소 속도 (절대 멈추지 않음)
        
        # [3] 게이트 인식
        self.check_dist = 1.5
        self.cluster_tol = 0.30
        self.entry_width_thresh = 0.35 # 몸통 진입 판단 기준
        self.exit_width_thresh = 0.45  # 탈출 판단 기준

        self.state = "IDLE"
        self.pass_timer = None
        self.last_steer = 0.0

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo("[M3 Potential] Ready. Logic: Push & Pull (No Reverse)")

    def mode_cb(self, msg):
        if msg.data == "LANE":
            self.state = "MONITORING"
            rospy.loginfo("[M3] Monitoring...")
        elif msg.data == "M3_RUN":
            if self.state not in ["PASS_INSIDE", "PASS_EXIT"]:
                self.state = "CENTERING"
                rospy.loginfo("[M3] Force Start -> CENTERING")
        elif msg.data == "M4_RUN":
            self.state = "IDLE"
            self.stop_robot()

    def stop_robot(self):
        self.pub_cmd.publish(Twist())

    def scan_cb(self, msg):
        if self.state == "IDLE": return

        # 데이터 전처리 (전방 180도 유효 데이터만)
        ranges = np.array(msg.ranges)
        angles = np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        valid = (ranges > 0.05) & (ranges < 4.0) & np.isfinite(ranges)
        r = ranges[valid]
        a = angles[valid]

        if len(r) == 0: return

        # === 좌우 측면 거리 측정 (상태 전환용) ===
        l_side = np.min(r[(a>60)&(a<120)]) if np.any((a>60)&(a<120)) else 2.0
        r_side = np.min(r[(a>-120)&(a<-60)]) if np.any((a>-120)&(a<-60)) else 2.0

        # =========================================
        # 1. 감시 (MONITORING)
        # =========================================
        if self.state == "MONITORING":
            # (기존 트리거 로직 유지 - 1.5m 내 갭 감지 시 시작)
            if self.check_trigger(r, a):
                self.pub_mode.publish("M3_RUN")
                self.state = "CENTERING"

        # =========================================
        # 2. 진입 및 주행 (CENTERING / PASS_INSIDE 통합 제어)
        # =========================================
        elif self.state in ["CENTERING", "PASS_INSIDE"]:
            
            # [상태 전환] CENTERING -> PASS_INSIDE
            if self.state == "CENTERING":
                if l_side < self.entry_width_thresh and r_side < self.entry_width_thresh:
                    rospy.logwarn(f"[M3] Body Inside -> PASS_INSIDE")
                    self.state = "PASS_INSIDE"

            # [상태 전환] PASS_INSIDE -> PASS_EXIT
            elif self.state == "PASS_INSIDE":
                if l_side > self.exit_width_thresh or r_side > self.exit_width_thresh:
                    rospy.logwarn(f"[M3] Exit Detected -> PASS_EXIT")
                    self.state = "PASS_EXIT"
                    self.pass_timer = rospy.Time.now()
                    return

            # --- [핵심] Push & Pull 주행 로직 ---
            
            # 1. Pull Vector (당기는 힘) 계산
            # - 게이트가 보이면 게이트 중심이 목표
            # - 안 보이면 '가장 멀리 뚫린 곳(Free Space)'이 목표
            gate_found, gate_dist, gate_ang = self.get_narrow_gate_target(r, a)
            
            if gate_found:
                target_angle = gate_ang
                # 게이트가 멀리 있으면 조향을 좀 약하게, 가까우면 강하게
                pull_strength = self.force_pull_gain
            else:
                # 게이트가 안 보일 땐, 전방 1.5m 내에서 가장 먼 곳(Max Range)을 향함
                # 이것이 자연스럽게 '통로 중앙'이나 '뚫린 곳'을 의미함
                mask_f = (a > -60) & (a < 60) & (r < 2.0)
                if np.any(mask_f):
                    # 거리가 먼 점들의 평균 각도 (너무 튀는 값 제외)
                    # 단순히 max()를 쓰면 노이즈에 취약하므로 상위 10% 거리의 평균 각도 사용
                    r_f = r[mask_f]
                    a_f = a[mask_f]
                    threshold = np.percentile(r_f, 90) # 상위 10% 거리
                    far_mask = r_f >= threshold
                    target_angle = np.mean(a_f[far_mask])
                else:
                    target_angle = 0.0 # 꽉 막혔으면 일단 직진 (Push가 알아서 피하게 함)
                
                pull_strength = self.force_pull_gain * 0.8 # 탐색 시엔 조금 약하게

            # 2. Push Vector (미는 힘) 계산
            # - 로봇 주변의 모든 장애물로부터 반발력 합산
            push_steer = 0.0
            closest_obs_dist = 2.0
            
            # 전방 120도 부채꼴 영역 검사
            mask_obs = (r < self.safe_margin) & (a > -90) & (a < 90)
            if np.any(mask_obs):
                obs_r = r[mask_obs]
                obs_a = a[mask_obs]
                closest_obs_dist = np.min(obs_r)

                # 각 장애물 점마다 반발력 생성
                # 힘의 크기 = (1 / 거리^2) -> 가까울수록 엄청 세짐
                # 방향 = 장애물 각도의 반대 (-sign)
                forces = (1.0 / (obs_r ** 2)) 
                
                # 왼쪽(+) 장애물은 오른쪽(-)으로 밀고, 오른쪽(-)은 왼쪽(+)으로 밈
                push_effects = -np.sign(obs_a) * forces
                
                # 전체 합산 후 정규화
                push_steer = np.sum(push_effects) 
                
                # 값이 너무 커지지 않게 클리핑 (부드러운 조향 위해)
                push_steer = np.clip(push_steer, -3.0, 3.0) * self.force_push_gain

            # 3. Force Blending (합력 조향)
            pull_steer = np.radians(target_angle) * pull_strength
            final_steer = pull_steer + push_steer
            final_steer = np.clip(final_steer, -1.5, 1.5)

            # 4. 속도 제어 (거리 비례, 절대 멈추지 않음)
            # 장애물이 가까울수록 속도를 줄이지만, min_speed 이하로는 안 떨어짐
            # 이렇게 해야 밀고 나갈 수 있음
            if closest_obs_dist < 0.5:
                # 0.2m일 때 min_speed, 0.5m일 때 base_speed
                ratio = (closest_obs_dist - 0.2) / 0.3
                speed = self.min_speed + (self.base_speed - self.min_speed) * np.clip(ratio, 0.0, 1.0)
            else:
                speed = self.base_speed

            self.drive_raw(speed, final_steer)

        # =========================================
        # 3. 완전 탈출 (PASS_EXIT)
        # =========================================
        elif self.state == "PASS_EXIT":
            elapsed = (rospy.Time.now() - self.pass_timer).to_sec()
            if elapsed < 1.5:
                # 부드럽게 직진
                self.drive_raw(0.2, self.last_steer * 0.5 * (1.0 - elapsed/1.5))
            else:
                self.finish()

    # === Helper Methods ===

    def check_trigger(self, r, a):
        # 1.5m 이내 평행/갭 트리거 (기존 로직 간소화)
        mask = (r < self.check_dist) & (a > -90) & (a < 90)
        r_m = r[mask]; a_m = a[mask]
        if len(r_m) < 10: return False
        
        rads = np.radians(a_m)
        points = np.stack((r_m * np.cos(rads), r_m * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > self.cluster_tol)[0] + 1
        clusters = np.split(points, split_idx)
        
        tips = []
        for clust in clusters:
            if len(clust) < 3: continue
            tips.append(clust[np.argmin(np.linalg.norm(clust, axis=1))])
            
        if len(tips) >= 2:
            for i in range(len(tips)):
                for j in range(i+1, len(tips)):
                    dist = np.linalg.norm(tips[i] - tips[j])
                    if 0.9 <= dist <= 1.5: return True
        return False

    def get_narrow_gate_target(self, r, a):
        # 1.5m 이내 게이트 탐색
        mask = (r < 1.5) & (a > -60) & (a < 60)
        r_g = r[mask]; a_g = a[mask]
        if len(r_g) < 5: return False, 0, 0

        rads = np.radians(a_g)
        points = np.stack((r_g * np.cos(rads), r_g * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > 0.3)[0] + 1
        clusters = np.split(np.arange(len(r_g)), split_idx)
        
        candidates = []
        valid_clusters = [c for c in clusters if len(c) >= 3]
        
        for i in range(len(valid_clusters) - 1):
            c_r = valid_clusters[i]; c_l = valid_clusters[i+1]
            pts_r = points[c_r]; pts_l = points[c_l]
            
            # 가장 가까운 점(Corner) 기준
            p1 = pts_r[np.argmin(np.linalg.norm(pts_r, axis=1))]
            p2 = pts_l[np.argmin(np.linalg.norm(pts_l, axis=1))]
            
            width = np.linalg.norm(p1 - p2)
            if 0.35 <= width <= 0.65:
                mid = (p1 + p2) / 2.0
                dist = np.linalg.norm(mid)
                ang = np.degrees(np.arctan2(mid[1], mid[0]))
                candidates.append((dist, ang))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return True, candidates[0][0], candidates[0][1]
        return False, 0, 0

    def drive_raw(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)
        self.last_steer = w

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