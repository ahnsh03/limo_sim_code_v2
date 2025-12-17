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

        # [1] 트리거 파라미터
        self.trigger_dist = 0.8      
        self.trigger_fov = 40.0      
        self.trigger_ratio = 0.2     

        # [2] 게이트 탐색 파라미터
        self.fov_limit = 100.0       
        self.search_range = 1.6       
        
        # [수정 1] 클러스터링 0.25m (튼튼하게 묶기)
        self.cluster_tol = 0.25       
        
        # [수정 2] 게이트 폭 범위 (1.3m 이내만 인정하여 벽 전체 인식 방지)
        self.target_gate_width = 0.60
        self.min_gate = 0.30         
        self.max_gate = 1.30          
        
        # [3] 주행 제어
        self.max_speed = 0.18
        self.kp_steer = 2.0          
        self.backup_speed = -0.15
        
        # [4] 통로 판단
        self.entry_thres = 0.35   
        self.exit_thres = 0.40    

        self.state = "IDLE"          
        self.trigger_start = None
        
        self.entry_start_time = None
        self.blind_entry_duration = 1.5 
        self.pass_timer = None
        
        self.last_target_angle = 0.0
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)
        
        rospy.Subscriber(self.mode_topic, String, self.mode_cb)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)

        rospy.loginfo("[M3_GATE] Ready. Logic: Outer Edges + Max Width 1.3m")
    
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
                self.state = "SEARCH"
                self.last_target_angle = 0.0 
                rospy.logwarn("[M3] State -> SEARCH Started")
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
        # 1. 감시 (MONITORING)
        # =========================================
        if self.state == "MONITORING":
            mask = (a >= -self.trigger_fov) & (a <= self.trigger_fov)
            r_chk = r[mask]
            if len(r_chk) > 0:
                ratio = np.sum(r_chk < self.trigger_dist) / float(len(r_chk))
                if ratio > self.trigger_ratio:
                    if self.trigger_start is None: self.trigger_start = rospy.Time.now()
                    elif (rospy.Time.now() - self.trigger_start).to_sec() > 0.1:
                        rospy.logwarn(f"[M3] Wall Detected (<{self.trigger_dist}m) -> SEARCH")
                        self.pub_mode.publish("M3_RUN")
                        self.state = "SEARCH"
                else:
                    self.trigger_start = None

        # =========================================
        # 2. 게이트 탐색 (SEARCH)
        # =========================================
        elif self.state == "SEARCH":
            # [Safety] 초근접 회피
            front_mask = (a > -40) & (a < 40)
            if np.any(front_mask):
                min_d = np.min(r[front_mask])
                if min_d < 0.20:
                    rospy.logwarn_throttle(0.5, f"[SAFETY] Too Close ({min_d:.2f}m)! Backing...")
                    min_idx = np.argmin(r[front_mask])
                    avoid_ang = a[front_mask][min_idx]
                    turn = -1.0 if avoid_ang > 0 else 1.0
                    self.drive_raw(-0.15, turn)
                    return

            # === 데이터 준비 (1.6m 이내) ===
            mask_search = (a >= -self.fov_limit) & (a <= self.fov_limit) & (r < self.search_range)
            r_search = r[mask_search]
            a_search = a[mask_search]
            
            best_angle = 0.0
            found_gate = False
            
            if len(r_search) >= 5:
                # 1. 좌표 변환
                rads = np.radians(a_search)
                xs = r_search * np.cos(rads)
                ys = r_search * np.sin(rads)

                # 2. 클러스터링 (0.25m 기준)
                diff = np.abs(np.diff(r_search))
                split_indices = np.where(diff > self.cluster_tol)[0] + 1
                clusters = np.split(np.arange(len(r_search)), split_indices)
                valid_clusters = [c for c in clusters if len(c) >= 3]

                gap_candidates = []
                
                # 3. 인접한 클러스터 분석
                for i in range(len(valid_clusters) - 1):
                    c_right = valid_clusters[i]   # 오른쪽 덩어리
                    c_left = valid_clusters[i+1]  # 왼쪽 덩어리
                    
                    # [핵심 수정] 사용자 요청 로직: Outer Edges (각도 극단값) 사용
                    # c_right[0]: 오른쪽 덩어리의 시작점 (각도가 가장 작은/오른쪽 끝)
                    # c_left[-1]: 왼쪽 덩어리의 끝점 (각도가 가장 큰/왼쪽 끝)
                    
                    idx_r = c_right[0]   # Rightmost of Right Cluster
                    idx_l = c_left[-1]   # Leftmost of Left Cluster

                    # 좌표 추출
                    p_r = np.array([xs[idx_r], ys[idx_r]]) 
                    p_l = np.array([xs[idx_l], ys[idx_l]]) 

                    # (A) Gap 너비
                    width = np.linalg.norm(p_r - p_l)

                    # (B) Gap 중심 각도
                    mid_point = (p_r + p_l) / 2.0
                    center_angle = np.degrees(np.arctan2(mid_point[1], mid_point[0]))

                    # [DEBUG]
                    # 여기서 1.3m 필터가 작동하여 "벽 전체"가 잡히는 것을 방지함
                    is_valid = (self.min_gate <= width <= self.max_gate)
                    
                    rospy.loginfo_throttle(0.5, 
                        f"[CHECK] OuterEdges Gap W:{width:.2f}m | Ang:{center_angle:.1f} | Valid:{is_valid}")

                    if is_valid:
                        gap_candidates.append((width, center_angle))

                # 4. 최적 Gap 선택
                if gap_candidates:
                    gap_candidates.sort(key=lambda x: abs(x[0] - self.target_gate_width)) 
                    best_angle = gap_candidates[0][1]
                    found_gate = True
                    rospy.logwarn(f"[M3] Locked GAP (W={gap_candidates[0][0]:.2f}, Ang={best_angle:.1f}) -> Entering")

            # === [행동 결정] ===
            if found_gate:
                self.drive(best_angle)
                self.last_target_angle = best_angle
                self.entry_start_time = rospy.Time.now()
                self.state = "PASS_ENTRY"
            else:
                # 못 찾았을 때: 회전하며 탐색
                if len(r_search) > 0:
                    min_idx = np.argmin(r_search)
                    anc_ang = a_search[min_idx]
                    
                    if abs(anc_ang) < 30: 
                        turn = -0.6 if anc_ang > 0 else 0.6
                        self.drive_raw(0.0, turn)
                    else: 
                        self.drive_raw(0.12, 0.0)
                else:
                    self.drive_raw(0.12, 0.0) 

        # =========================================
        # 3. 진입 중 (PASS_ENTRY) - Blind Mode
        # =========================================
        elif self.state == "PASS_ENTRY":
            elapsed = (rospy.Time.now() - self.entry_start_time).to_sec()
            
            # 1.5초간은 찾은 각도로 직진
            if elapsed < self.blind_entry_duration:
                rospy.loginfo_throttle(0.5, f"[ENTRY] Blind Drive... (Target: {self.last_target_angle:.1f})")
                self.drive(self.last_target_angle)
                return

            l_d = np.min(r[(a>60)&(a<100)]) if np.any((a>60)&(a<100)) else 3.0
            r_d = np.min(r[(a>-100)&(a<-60)]) if np.any((a>-100)&(a<-60)) else 3.0
            
            if l_d < 1.2 and r_d < 1.2:
                self.keep_center(l_d, r_d)
            else:
                self.drive_raw(0.15, 0.0)
            
            if l_d < self.entry_thres and r_d < self.entry_thres:
                rospy.logwarn(f"[M3] Inside (L={l_d:.2f}, R={r_d:.2f}) -> PASS_INSIDE")
                self.state = "PASS_INSIDE"

        # =========================================
        # 4. 통로 내부 (PASS_INSIDE)
        # =========================================
        elif self.state == "PASS_INSIDE":
            l_d = np.min(r[(a>60)&(a<100)]) if np.any((a>60)&(a<100)) else 2.0
            r_d = np.min(r[(a>-100)&(a<-60)]) if np.any((a>-100)&(a<-60)) else 2.0
            
            self.keep_center(l_d, r_d)
            
            if l_d > self.exit_thres or r_d > self.exit_thres:
                rospy.logwarn(f"[M3] Exiting (L={l_d:.2f}, R={r_d:.2f}) -> PASS_EXIT")
                self.state = "PASS_EXIT"
                self.pass_timer = rospy.Time.now()

        # =========================================
        # 5. 탈출 및 종료 (PASS_EXIT)
        # =========================================
        elif self.state == "PASS_EXIT":
            if (rospy.Time.now() - self.pass_timer).to_sec() < 0.8:
                self.drive_raw(0.15, 0.0) 
            else:
                self.finish()

    def keep_center(self, l_d, r_d):
        err = l_d - r_d
        steer = np.clip(err * 2.5, -1.5, 1.5)
        if l_d < 0.15: steer = -2.0
        elif r_d < 0.15: steer = 2.0
        self.drive_raw(0.15, steer)

    def drive(self, deg, speed=None):
        cmd = Twist()
        cmd.linear.x = speed if speed else self.max_speed
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
                rospy.loginfo("[M3_GATE] Shutting down, stopping robot...")
                node.stop_robot()
        
        def signal_handler(signum, frame):
            cleanup()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup)
        
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[M3_GATE] Interrupted by user")
        if node is not None:
            node.stop_robot()
    except rospy.ROSInterruptException:
        if node is not None:
            node.stop_robot()
    except Exception as e:
        rospy.logerr(f"[M3_GATE] Error: {e}")
        if node is not None:
            node.stop_robot()
    finally:
        if node is not None:
            node.stop_robot()