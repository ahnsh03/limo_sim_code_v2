#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import signal
import atexit
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

        # [1] 트리거 파라미터 (장애물 감지)
        self.trigger_dist = 1.0      # 1m 이내 감지 시 시작
        self.trigger_fov = 40.0      
        self.trigger_ratio = 0.2     

        # [2] 넓은 게이트(입구) 파라미터
        self.wide_search_range = 1.7  # 넓은 범위 탐색
        self.wide_cluster_tol = 0.2   # 0.2m 기준 클러스터링
        self.wide_min = 0.9           # 1.0 ~ 1.3m (여유값 0.9~1.4)
        self.wide_max = 1.4           
        
        # [3] 좁은 게이트(출구) 파라미터
        self.narrow_search_range = 1.6
        self.narrow_min = 0.30
        self.narrow_max = 0.65
        self.narrow_fov = 40.0        # 전방 +-40도

        # [4] 제어 파라미터
        self.kp_steer = 2.0           # 진입 조향 게인
        self.kp_align = 2.0           # 평행 맞추기 게인
        self.kp_center = 2.5          # 중앙 맞추기 게인
        
        self.front_safe_margin = 0.6  # 전방 장애물 최소 거리 (이보다 가까우면 후진)
        self.front_scan_dist = 1.2    # 전방 탐색 시작 거리 (이보다 멀면 전진)

        self.state = "IDLE"          
        self.trigger_start = None
        self.entry_start_time = None
        self.last_target_angle = 0.0
        
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo("[M3] Ready. 2-Stage: Wide Gate(1.0-1.3m) -> Center/Reverse -> Narrow Gate(0.3-0.6m)")
    
    def stop_robot(self):
        """로봇 정지"""
        twist = Twist()
        self.pub_cmd.publish(twist)

    def mode_cb(self, msg):
        mode = msg.data
        if mode == "LANE":
            self.state = "MONITORING"
            self.trigger_start = None
            rospy.loginfo("[M3] State -> MONITORING")
        elif mode == "M3_RUN":
            if "PASS" not in self.state:
                # 이미 진입 중이 아니면 넓은 게이트 찾기부터 시작
                self.state = "ENTRY_ALIGN"
                rospy.logwarn("[M3] State -> ENTRY_ALIGN (Find Wide Gate)")
        elif mode == "M4_RUN":
            self.state = "IDLE"
            rospy.loginfo("[M3] State -> IDLE")

    def scan_cb(self, msg):
        if self.state == "IDLE": return

        ranges = np.array(msg.ranges)
        angles = np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        valid = (ranges > 0.05) & (ranges < 4.0) & np.isfinite(ranges)
        r = ranges[valid]
        a = angles[valid]

        if len(r) == 0: return

        # =========================================
        # 1. 감시 (MONITORING) - 차선 주행 중
        # =========================================
        if self.state == "MONITORING":
            mask = (a >= -self.trigger_fov) & (a <= self.trigger_fov)
            r_chk = r[mask]
            if len(r_chk) > 0:
                # 1m 이내 장애물 비율 확인
                ratio = np.sum(r_chk < self.trigger_dist) / float(len(r_chk))
                if ratio > self.trigger_ratio:
                    if self.trigger_start is None: self.trigger_start = rospy.Time.now()
                    elif (rospy.Time.now() - self.trigger_start).to_sec() > 0.1:
                        rospy.logwarn(f"[M3] Obstacle Detected (<{self.trigger_dist}m) -> ENTRY_ALIGN")
                        self.pub_mode.publish("M3_RUN")
                        self.state = "ENTRY_ALIGN"
                else:
                    self.trigger_start = None

        # =========================================
        # 2. 넓은 게이트 진입 (ENTRY_ALIGN)
        # =========================================
        elif self.state == "ENTRY_ALIGN":
            # 1.7m 이내 데이터 사용
            mask = r < self.wide_search_range
            r_s = r[mask]
            a_s = a[mask]
            
            if len(r_s) < 5: 
                self.drive_raw(0.1, 0.0)
                return

            # 좌표 변환
            rads = np.radians(a_s)
            xs = r_s * np.cos(rads)
            ys = r_s * np.sin(rads)
            points = np.stack((xs, ys), axis=1)

            # 클러스터링 (0.2m 기준)
            diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
            split_idx = np.where(diffs > self.wide_cluster_tol)[0] + 1
            clusters = np.split(np.arange(len(r_s)), split_idx)
            valid_clusters = [c for c in clusters if len(c) >= 5]

            found_wide_gate = False
            best_ang = 0.0

            # 인접한 두 클러스터 사이 거리 분석
            for i in range(len(valid_clusters) - 1):
                c_right = valid_clusters[i]
                c_left = valid_clusters[i+1]

                # 각 클러스터에서 로봇과 가장 가까운 점(Tip) 찾기
                # (벽의 시작점일 확률이 높음)
                idx_r = c_right[np.argmin(r_s[c_right])]
                idx_l = c_left[np.argmin(r_s[c_left])]

                p1 = np.array([xs[idx_r], ys[idx_r]])
                p2 = np.array([xs[idx_l], ys[idx_l]])
                
                width = np.linalg.norm(p1 - p2)
                
                # 폭 1.0 ~ 1.3m 인지 확인
                if self.wide_min <= width <= self.wide_max:
                    mid_point = (p1 + p2) / 2.0
                    best_ang = np.degrees(np.arctan2(mid_point[1], mid_point[0]))
                    
                    found_wide_gate = True
                    rospy.logwarn_throttle(0.5, f"[ENTRY] Wide Gate Found: W={width:.2f} Ang={best_ang:.1f}")
                    break
            
            if found_wide_gate:
                # 게이트가 정면(-20~20도)에 오고, 거리가 가까워지면(0.8m) CENTERING으로 전환
                min_dist = np.min(r_s)
                if abs(best_ang) < 20 and min_dist < 0.8:
                    rospy.logwarn("[M3] Entering Walls -> Switch to CENTERING")
                    self.state = "CENTERING"
                else:
                    # 게이트 중앙을 향해 진입 (수직 진입 유도)
                    self.drive(best_ang)
            else:
                # 못 찾았으면 천천히 직진/탐색
                self.drive_raw(0.12, -0.2) # 우측에 있을 확률이 높으므로 살짝 우회전 탐색

        # =========================================
        # 3. 벽 사이 중앙 맞추기 & 전후진 (CENTERING)
        # =========================================
        elif self.state == "CENTERING":
            # 1. 왼쪽 벽 / 오른쪽 벽 직선 검출
            # 왼쪽: 45~110도, 오른쪽: -110~-45도 (전방 장애물 제외하고 측면만 봄)
            mask_l = (a > 45) & (a < 110) & (r < 1.0)
            mask_r = (a > -110) & (a < -45) & (r < 1.0)
            
            # 전방 장애물 거리 확인 (-30 ~ 30도)
            mask_front = (a > -30) & (a < 30) & (r < 1.5)
            front_dist = np.min(r[mask_front]) if np.any(mask_front) else 2.0

            # (A) 헤딩 및 중앙 정렬 조향 계산
            steer = 0.0
            
            # 왼쪽 벽 기준 오차
            l_dist = 0.6 # 기본값
            if np.sum(mask_l) > 5:
                # y = mx + c 근사 (로봇 좌표계)
                # X축(전방) 기준으로 Fitting
                rads = np.radians(a[mask_l])
                lx = r[mask_l] * np.cos(rads)
                ly = r[mask_l] * np.sin(rads)
                
                try:
                    slope, intercept = np.polyfit(lx, ly, 1) # slope가 0이어야 평행
                    l_dist = abs(intercept)
                    
                    # 평행 제어 (기울기) + 거리 제어 (0.6m)
                    steer += (slope * self.kp_align) + ((l_dist - 0.6) * self.kp_center)
                except: pass

            # 오른쪽 벽 기준 오차 (보조)
            if np.sum(mask_r) > 5:
                rads = np.radians(a[mask_r])
                rx = r[mask_r] * np.cos(rads)
                ry = r[mask_r] * np.sin(rads)
                try:
                    slope, intercept = np.polyfit(rx, ry, 1)
                    r_dist = abs(intercept)
                    # 오른쪽 벽은 y가 음수. slope도 반대.
                    # 거리가 0.6보다 작으면(가까우면) 왼쪽으로(Steer +)
                    steer += (slope * self.kp_align) - ((r_dist - 0.6) * self.kp_center)
                except: pass
            
            steer = np.clip(steer, -1.5, 1.5)

            # (B) 전후진 로직 (공간 확보)
            speed = 0.0
            log_act = "STOP"

            if front_dist < self.front_safe_margin: # 0.6m 보다 가까우면 후진
                speed = -0.15
                log_act = "BACK"
            elif front_dist > self.front_scan_dist: # 1.2m 보다 멀면 전진
                speed = 0.12
                log_act = "FWD"
            else:
                # 적당한 위치 (0.6 ~ 1.2m 사이) -> 정지 후 좁은 게이트 탐색
                speed = 0.0
                log_act = "SCAN"
                
                # 정렬이 안정적일 때 좁은 게이트 탐색
                if self.find_narrow_gate(r, a): return

            rospy.loginfo_throttle(0.5, f"[{log_act}] Front:{front_dist:.2f} Steer:{steer:.2f}")
            self.drive_raw(speed, steer)

        # =========================================
        # 4. 좁은 게이트 진입 (PASS_ENTRY)
        # =========================================
        elif self.state == "PASS_ENTRY":
            elapsed = (rospy.Time.now() - self.entry_start_time).to_sec()
            
            if elapsed < 1.5:
                self.drive(self.last_target_angle)
                return

            l_d = np.min(r[(a>60)&(a<100)]) if np.any((a>60)&(a<100)) else 3.0
            r_d = np.min(r[(a>-100)&(a<-60)]) if np.any((a>-100)&(a<-60)) else 3.0
            
            if l_d < 1.0 and r_d < 1.0:
                self.keep_center(l_d, r_d)
            else:
                self.drive_raw(0.15, 0.0)
            
            if l_d < 0.4 and r_d < 0.4: # 충분히 통과했으면 내부 로직으로
                rospy.logwarn(f"[M3] Inside -> PASS_INSIDE")
                self.state = "PASS_INSIDE"

        # =========================================
        # 5. 통로 내부 (PASS_INSIDE)
        # =========================================
        elif self.state == "PASS_INSIDE":
            l_d = np.min(r[(a>60)&(a<100)]) if np.any((a>60)&(a<100)) else 2.0
            r_d = np.min(r[(a>-100)&(a<-60)]) if np.any((a>-100)&(a<-60)) else 2.0
            
            self.keep_center(l_d, r_d)
            
            # 출구 감지 (옆이 뚫리면)
            if l_d > 0.6 or r_d > 0.6:
                rospy.logwarn(f"[M3] Exiting -> PASS_EXIT")
                self.state = "PASS_EXIT"
                self.pass_timer = rospy.Time.now()

        # =========================================
        # 6. 탈출 (PASS_EXIT)
        # =========================================
        elif self.state == "PASS_EXIT":
            if (rospy.Time.now() - self.pass_timer).to_sec() < 1.0:
                self.drive_raw(0.2, 0.0) 
            else:
                self.finish()

    def find_narrow_gate(self, r, a):
        """ 정렬 상태에서 전방 좁은 게이트(0.3~0.6m) 탐색 """
        mask = (r < self.narrow_search_range) & (a > -self.narrow_fov) & (a < self.narrow_fov)
        r_g = r[mask]
        a_g = a[mask]
        
        if len(r_g) < 5: return False

        # 좌표 변환
        rads = np.radians(a_g)
        xs = r_g * np.cos(rads)
        ys = r_g * np.sin(rads)
        points = np.stack((xs, ys), axis=1)

        # 클러스터링 (0.2m)
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_idx = np.where(diffs > 0.2)[0] + 1
        clusters = np.split(np.arange(len(r_g)), split_idx)
        valid_clusters = [c for c in clusters if len(c) >= 3]

        for i in range(len(valid_clusters) - 1):
            c_r = valid_clusters[i]
            c_l = valid_clusters[i+1]
            
            # Inner Edges
            idx_r = c_r[-1]
            idx_l = c_l[0]

            p1 = np.array([xs[idx_r], ys[idx_r]])
            p2 = np.array([xs[idx_l], ys[idx_l]])
            
            width = np.linalg.norm(p1 - p2)
            
            # 폭 조건 (0.3 ~ 0.65m)
            if self.narrow_min <= width <= self.narrow_max:
                # 수직 조건 (X좌표 차이가 작아야 함)
                if abs(p1[0] - p2[0]) < 0.2:
                    mid_point = (p1 + p2) / 2.0
                    target_ang = np.degrees(np.arctan2(mid_point[1], mid_point[0]))
                    
                    rospy.logwarn(f"[M3] EXIT FOUND! W:{width:.2f} Ang:{target_ang:.1f}")
                    self.last_target_angle = target_ang
                    self.entry_start_time = rospy.Time.now()
                    self.state = "PASS_ENTRY"
                    return True
        return False

    def keep_center(self, l_d, r_d):
        err = l_d - r_d
        steer = np.clip(err * 2.5, -1.5, 1.5)
        if l_d < 0.15: steer = -2.0
        elif r_d < 0.15: steer = 2.0
        self.drive_raw(0.15, steer)

    def drive(self, deg, speed=None):
        cmd = Twist()
        cmd.linear.x = speed if speed else 0.15 # 진입 시 속도
        cmd.angular.z = np.radians(deg) * self.kp_steer
        if abs(cmd.angular.z) > 0.5: cmd.linear.x *= 0.5
        self.pub_cmd.publish(cmd)

    def drive_raw(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

    def finish(self):
        self.pub_cmd.publish(Twist())
        rospy.logwarn("[M3] Mission Complete -> Trigger M4")
        self.pub_mode.publish("M4_RUN")
        self.state = "IDLE"

if __name__ == "__main__":
    node = None
    try:
        node = M3GateNav()
        
        # 종료 시 자동 멈춤 핸들러 등록
        def cleanup():
            if node is not None:
                rospy.loginfo("[M3] Shutting down, stopping robot...")
                node.stop_robot()
        
        def signal_handler(signum, frame):
            cleanup()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup)
        
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[M3] Interrupted by user")
        if node is not None:
            node.stop_robot()
    except rospy.ROSInterruptException:
        if node is not None:
            node.stop_robot()
    except Exception as e:
        rospy.logerr(f"[M3] Error: {e}")
        if node is not None:
            node.stop_robot()
    finally:
        if node is not None:
            node.stop_robot()