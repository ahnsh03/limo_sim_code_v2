#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String

class M3GateNav:
    def __init__(self):
        rospy.init_node("m3_gate_nav", anonymous=False)

        # === 설정 ===
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/limo/mode")
        self.cam_topic = rospy.get_param("~cam_topic", "/camera/rgb/image_raw") 

        # [1] 거리 및 인식 파라미터
        self.check_dist = 1.5         
        self.gate_lock_dist = 1.3     
        self.cluster_tol = 0.30       
        
        # [2] 포텐셜 필드 파라미터
        self.force_pull_gain = 3.0    
        self.force_push_gain = 1.0    
        self.safe_margin = 0.6        
        
        # [3] 주행 제어 파라미터
        self.base_speed = 0.22
        self.steer_limit = 2.0        
        self.last_steer = 0.0
        self.max_steer_change = 2.00 
        
        # [4] 상태 전환 임계값
        self.entry_dist_thresh = 0.40   
        self.exit_dist_thresh = 0.45    

        self.state = "IDLE"
        self.pass_timer = None
        self.entry_stable_start = None
        
        # [Memory Variables]
        self.target_lost_time = None
        self.last_target_dist = 0.0
        self.last_target_angle = 0.0

        self.bridge = CvBridge()
        self.cv_image = None
        self.cam_fov = 60.0  
        self.img_width = 640 

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)
        rospy.Subscriber(self.cam_topic, Image, self.image_cb)

        rospy.loginfo(f"[M3 Final] Ready. Blind Push Logic Applied.")

    def mode_cb(self, msg):
        if msg.data == "LANE":
            self.state = "MONITORING"
            rospy.loginfo("[State] MONITORING (Gap Search)")
        elif msg.data == "M3_RUN":
            if self.state not in ["PASS_INSIDE", "PASS_EXIT", "DELIVERY_M4"]:
                self.state = "CENTERING"
                self.entry_stable_start = None
                rospy.loginfo("[State] Force Start -> CENTERING")
        elif msg.data == "M4_RUN":
            self.state = "IDLE"
            self.stop_robot()

    def image_cb(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.img_width = self.cv_image.shape[1]
        except CvBridgeError as e:
            pass

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

        l_mask = (a > 60) & (a < 120) & (r < 1.0)
        r_mask = (a > -120) & (a < -60) & (r < 1.0)
        l_side = np.min(r[l_mask]) if np.any(l_mask) else 2.0
        r_side = np.min(r[r_mask]) if np.any(r_mask) else 2.0

        # =========================================
        # 1. 감시 (MONITORING)
        # =========================================
        if self.state == "MONITORING":
            if self.check_trigger(r, a):
                self.pub_mode.publish("M3_RUN")
                self.state = "CENTERING"
                self.entry_stable_start = None
                rospy.logwarn(f"[State] CENTERING (Triggered by Gap)")

        # =========================================
        # 2. 접근 및 정렬 (CENTERING)
        # =========================================
        elif self.state == "CENTERING":
            if l_side < self.entry_dist_thresh and r_side < self.entry_dist_thresh:
                if self.entry_stable_start is None:
                    self.entry_stable_start = rospy.Time.now()
                else:
                    elapsed = (rospy.Time.now() - self.entry_stable_start).to_sec()
                    if elapsed > 0.5: 
                        rospy.logwarn(f"[State] PASS_INSIDE (Entry Confirmed!)")
                        self.state = "PASS_INSIDE"
                        return
            else:
                self.entry_stable_start = None 

            gate_found, gate_dist, gate_ang = self.find_gate_target(r, a, max_dist=self.gate_lock_dist)

            target_steer = 0.0
            if gate_found:
                self.safe_margin = 0.3
                pull = np.radians(gate_ang) * self.force_pull_gain
                push = self.calculate_potential_push(r, a)
                target_steer = pull + push
            else:
                self.safe_margin = 0.6
                pull_angle = self.find_free_space(r, a)
                pull = np.radians(pull_angle) * (self.force_pull_gain * 0.8)
                push = self.calculate_potential_push(r, a)
                target_steer = pull + push

            self.drive_smooth_physics(r, a, target_steer)

        # =========================================
        # 3. 통로 내부 (PASS_INSIDE)
        # =========================================
        elif self.state == "PASS_INSIDE":
            self.safe_margin = 0.3
            
            if l_side > self.exit_dist_thresh or r_side > self.exit_dist_thresh:
                rospy.logwarn(f"[State] PASS_EXIT (Exit Detected!)")
                self.state = "PASS_EXIT"
                self.pass_timer = rospy.Time.now()
                return

            gate_found, _, gate_ang = self.find_gate_target(r, a, max_dist=1.5)
            push = self.calculate_potential_push(r, a)

            if gate_found:
                pull = np.radians(gate_ang) * self.force_pull_gain
                target_steer = pull + push
            else:
                pull_angle = self.find_free_space(r, a)
                pull = np.radians(pull_angle) * (self.force_pull_gain * 0.8)
                target_steer = pull + push
            
            self.drive_smooth_physics(r, a, target_steer)

        # =========================================
        # 4. 탈출 (PASS_EXIT)
        # =========================================
        elif self.state == "PASS_EXIT":
            elapsed = (rospy.Time.now() - self.pass_timer).to_sec()
            if elapsed < 1.0:
                self.drive_raw(0.2, 0.0) 
            else:
                rospy.logwarn("[State] DELIVERY_M4 (Start Delivery)")
                self.state = "DELIVERY_M4"
                self.target_lost_time = None 

        # =========================================
        # 5. M4 진입점 배달 (DELIVERY_M4) - [Fixed Logic]
        # =========================================
        elif self.state == "DELIVERY_M4":
            found, dist, ang = self.find_cone_gate_center(r, a)
            
            # [A] 찾았을 때
            if found:
                self.last_target_dist = dist
                self.last_target_angle = ang
                self.target_lost_time = None 
                
                target_dist = dist - 0.3 
                rospy.loginfo_throttle(0.5, f"[DELIVERY] Found: {ang:.1f}deg | Rem: {target_dist:.2f}m")

                # 도착 판단
                if abs(target_dist) < 0.1:
                    rospy.logwarn("[State] COMPLETE -> M4_RUN")
                    self.finish() 
                    return
                
                kp_dist = 0.6
                kp_ang = 1.5
                v_calc = target_dist * kp_dist
                
                # [수정] 최소 속도 0.12 보장 (정지 방지)
                v = np.clip(v_calc, 0.12, 0.22)
                w = np.clip(np.radians(ang) * kp_ang, -1.0, 1.0)
                self.drive_raw(v, w)

            # [B] 놓쳤을 때 (Memory)
            else:
                if self.target_lost_time is None:
                    self.target_lost_time = rospy.Time.now()
                
                elapsed_lost = (rospy.Time.now() - self.target_lost_time).to_sec()
                
                if elapsed_lost < 1.0 and self.last_target_dist > 0:
                    rospy.loginfo_throttle(0.5, f"[DELIVERY] Blind Push... ({elapsed_lost:.1f}s)")
                    
                    # [수정] 거리 계산 없이 고정 속도로 밀고 들어감
                    # 이유: 멈추지 않게 하기 위함. 방향은 마지막 각도 유지.
                    v = 0.15 
                    w = np.clip(np.radians(self.last_target_angle) * 1.5, -1.0, 1.0)
                    self.drive_raw(v, w)
                else:
                    # 1초 이상 놓치면 탐색 모드
                    rospy.loginfo_throttle(1.0, "[DELIVERY] Lost too long.. Scanning..")
                    self.drive_raw(0.12, 0.0) # 탐색 시에도 0.12 유지

    # === Smoothing Logic ===
    def drive_smooth_physics(self, r, a, target_w):
        mask_f = (a > -15) & (a < 15) & (r < 1.0)
        min_front = np.min(r[mask_f]) if np.any(mask_f) else 2.0
        
        if min_front < 0.18: speed = -0.15
        elif min_front < 0.8: speed = 0.12
        else: speed = self.base_speed

        target_w = np.clip(target_w, -self.steer_limit, self.steer_limit)
        delta = target_w - self.last_steer
        clamped_delta = np.clip(delta, -self.max_steer_change, self.max_steer_change)
        smooth_w = self.last_steer + clamped_delta
        
        self.drive_raw(speed, smooth_w)

    # === Helper Methods ===
    def drive_raw(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)
        self.last_steer = w

    def check_trigger(self, r, a):
        mask = (r < self.check_dist) & (a > -90) & (a < 90)
        r_m = r[mask]; a_m = a[mask]
        if len(r_m) < 5: return False 
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

    def calculate_potential_push(self, r, a):
        push_steer = 0.0
        mask_obs = (r < self.safe_margin) & (a > -90) & (a < 90)
        if np.any(mask_obs):
            obs_r = r[mask_obs]
            obs_a = a[mask_obs]
            forces = (1.0 / (obs_r ** 2)) 
            push_effects = -np.sign(obs_a) * forces
            push_steer = np.sum(push_effects) 
            push_steer = np.clip(push_steer, -3.0, 3.0) * self.force_push_gain
        return push_steer

    def find_free_space(self, r, a):
        mask = (a > -60) & (a < 60) & (r < 2.0)
        if np.any(mask):
            threshold = np.percentile(r[mask], 90)
            target_angle = np.mean(a[mask][r[mask] >= threshold])
            return target_angle
        return 0.0

    def find_gate_target(self, r, a, max_dist):
        mask = (r < max_dist) & (a > -50) & (a < 50)
        r_g = r[mask]; a_g = a[mask]
        if len(r_g) < 5: return False, 0, 0
        rads = np.radians(a_g)
        points = np.stack((r_g * np.cos(rads), r_g * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > self.cluster_tol)[0] + 1
        clusters = np.split(np.arange(len(r_g)), split_idx)
        candidates = []
        valid_clusters = [c for c in clusters if len(c) >= 3]
        for i in range(len(valid_clusters) - 1):
            c_r = valid_clusters[i]; c_l = valid_clusters[i+1]
            pts_r = points[c_r]; pts_l = points[c_l]
            p1 = pts_r[np.argmin(np.linalg.norm(pts_r, axis=1))]
            p2 = pts_l[np.argmin(np.linalg.norm(pts_l, axis=1))]
            width = np.linalg.norm(p1 - p2)
            if 0.3 <= width <= 0.7:
                mid = (p1 + p2) / 2.0
                dist = np.linalg.norm(mid)
                ang = np.degrees(np.arctan2(mid[1], mid[0]))
                candidates.append((dist, ang))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return True, candidates[0][0], candidates[0][1]
        return False, 0, 0

    def is_red_object(self, lidar_pt):
        if self.cv_image is None: return False 
        x, y = lidar_pt[0], lidar_pt[1]
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad) 
        if abs(angle_deg) > (self.cam_fov / 2.0): return False 
        ratio = 0.5 - (angle_deg / self.cam_fov)
        px_x = int(ratio * self.img_width)
        if px_x < 0 or px_x >= self.img_width: return False
        h, w, _ = self.cv_image.shape
        roi_y_start = int(h * 0.5) 
        roi_y_end = int(h * 0.9)   
        x_start = max(0, px_x - 10)
        x_end = min(w, px_x + 10)
        roi = self.cv_image[roi_y_start:roi_y_end, x_start:x_end]
        if roi.size == 0: return False
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 50]); upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)
        return (np.count_nonzero(mask) / mask.size) > 0.1

    def find_cone_gate_center(self, r, a):
        mask = (r < 1.9) & (a > -90) & (a < 90)
        r_c = r[mask]; a_c = a[mask]
        if len(r_c) < 3: return False, 0, 0
        rads = np.radians(a_c)
        points = np.stack((r_c * np.cos(rads), r_c * np.sin(rads)), axis=1)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > 0.05)[0] + 1 
        clusters = np.split(points, split_idx)
        cone_candidates = []
        for clust in clusters:
            if len(clust) < 2: continue 
            width = np.linalg.norm(clust[0] - clust[-1])
            if width < 0.30: 
                center = np.mean(clust, axis=0)
                cx, cy = center[0], center[1]
                angle_deg = np.degrees(np.arctan2(cy, cx))
                if abs(angle_deg) < (self.cam_fov / 2.0):
                    if self.is_red_object(center): cone_candidates.append(center)
                else: cone_candidates.append(center)
        if len(cone_candidates) >= 2:
            cone_candidates.sort(key=lambda p: np.linalg.norm(p))
            c1 = cone_candidates[0]; c2 = cone_candidates[1]
            gate_width = np.linalg.norm(c1 - c2)
            if 0.4 <= gate_width <= 1.5:
                mid_point = (c1 + c2) / 2.0
                dist = np.linalg.norm(mid_point)
                ang = np.degrees(np.arctan2(mid_point[1], mid_point[0]))
                return True, dist, ang
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
    except rospy.ROSInterruptException:
        pass