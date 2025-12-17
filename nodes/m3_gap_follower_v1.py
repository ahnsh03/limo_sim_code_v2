#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3 갭 중심 추종 컨트롤러 (경로별 갭 탐색 + 통로 중앙 주행, 로봇 차폭/안전여유 반영)

핵심 동작:
- LEFT / CENTER / RIGHT 세 구간 중 가장 "열린" 경로를 한 번만 선택(경로 잠금)
- 선택된 경로 안에서 가장 큰 갭을 찾아 갭 중심 방향으로 접근 (stage=approach)
- 통로 진입 판단 시 corridor 단계로 전환 후 중앙 유지 주행
- 통로 통과 완료 시 정지 + /m3_auto_trigger/set_trigger(False)로 트리거 리셋(옵션)

최적화/보강:
- gap_split_deg(갭 분리 기준 각도) 파라미터화
- debug 플래그로 과도한 로그 억제(제트슨 부담 완화)
- backoff_omega 파라미터화
"""

import rospy
import math
import numpy as np
import signal
import atexit
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool


class M3GapFollower:
    def __init__(self):
        rospy.init_node("m3_gap_follower", anonymous=False)

        rospy.logwarn("[M3GapFollower] ### VERSION optimized-v16+ (debug-toggle + gap_split_deg + backoff_omega) ###")

        # Debug
        self.debug = bool(rospy.get_param("~debug", False))

        # === 로봇(리모) 기하 파라미터 ===
        self.robot_width = rospy.get_param("~robot_width", 0.22)
        self.side_safety_margin = rospy.get_param("~side_safety_margin", 0.05)
        self.safe_side_clearance = self.robot_width / 2.0 + self.side_safety_margin

        self.outward_angle_bias_deg = rospy.get_param("~outward_angle_bias_deg", 1.0)
        self.max_approach_angle_deg = rospy.get_param("~max_approach_angle_deg", 35.0)

        # === 토픽 파라미터 ===
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        # 각도 범위 (경로 선택용)
        self.right_angle_min = rospy.get_param("~right_angle_min", -60.0)
        self.right_angle_max = rospy.get_param("~right_angle_max", -25.0)
        self.center_angle_min = rospy.get_param("~center_angle_min", -30.0)
        self.center_angle_max = rospy.get_param("~center_angle_max", -5.0)
        self.left_angle_min = rospy.get_param("~left_angle_min", -10.0)
        self.left_angle_max = rospy.get_param("~left_angle_max", 20.0)

        self.gap_search_margin_deg = rospy.get_param("~gap_search_margin_deg", 5.0)
        self.gap_track_window_deg = rospy.get_param("~gap_track_window_deg", 20.0)
        self.gap_center_smoothing = rospy.get_param("~gap_center_smoothing", 0.3)

        # 갭 분리 기준(원본 하드코딩 5°)
        self.gap_split_deg = rospy.get_param("~gap_split_deg", 5.0)

        # 열린 공간 판단 범위 (경로 선택용)
        self.open_space_range_min = rospy.get_param("~open_space_range_min", 1.2)
        self.open_space_range_max = rospy.get_param("~open_space_range_max", 4.0)

        # 갭 찾기 파라미터 (실제 주행용)
        self.gap_detect_range_min = rospy.get_param("~gap_detect_range_min", 0.05)
        self.gap_detect_range_max = rospy.get_param("~gap_detect_range_max", 4.0)
        self.min_gap_width_deg = rospy.get_param("~min_gap_width_deg", 1.0)
        self.gap_margin_deg = rospy.get_param("~gap_margin_deg", 2.0)

        # 속도 파라미터
        self.v_nominal = rospy.get_param("~v_nominal", 0.20)
        self.v_min = rospy.get_param("~v_min", 0.12)
        self.omega_max = rospy.get_param("~omega_max", 1.5)
        self.omega_gain = rospy.get_param("~omega_gain", 2.0)

        # 벽/장애물 인식 및 중앙 유지 파라미터
        self.wall_follow_enabled = rospy.get_param("~wall_follow_enabled", True)
        self.corridor_width = rospy.get_param("~corridor_width", 0.425)
        self.wall_critical_dist = rospy.get_param("~wall_critical_dist", 0.15)
        self.center_balance_gain = rospy.get_param("~center_balance_gain", 2.5)

        # 정면(center) 장애물 회피 파라미터
        self.center_avoid_dist = rospy.get_param("~center_avoid_dist", 0.20)
        self.center_avoid_gain = rospy.get_param("~center_avoid_gain", 2.0)

        # 통로 입구 근처 중앙 정렬(approach 단계용)
        self.pre_corridor_center_dist = rospy.get_param("~pre_corridor_center_dist", 0.8)
        self.pre_center_balance_gain = rospy.get_param("~pre_center_balance_gain", 1.5)

        # 통로 진입 / 통과 판정 관련 파라미터
        default_corridor_min = self.safe_side_clearance
        default_corridor_max = self.safe_side_clearance + 0.25

        self.corridor_side_min = rospy.get_param("~corridor_side_min", default_corridor_min)
        self.corridor_side_max = rospy.get_param("~corridor_side_max", default_corridor_max)
        self.corridor_side_diff_max = rospy.get_param("~corridor_side_diff_max", 0.25)
        self.corridor_front_max = rospy.get_param("~corridor_front_max", 1.2)

        self.pass_front_clear_dist = rospy.get_param("~pass_front_clear_dist", 1.0)
        self.corridor_clear_margin = rospy.get_param("~corridor_clear_margin", 0.30)
        self.pass_complete_time = rospy.get_param("~pass_complete_time", 0.5)

        # EMERGENCY 백오프 설정
        self.emergency_backoff_enabled = rospy.get_param("~emergency_backoff_enabled", True)
        self.emergency_backoff_dist = rospy.get_param("~emergency_backoff_dist", 0.14)
        self.emergency_backoff_speed = rospy.get_param("~emergency_backoff_speed", 0.12)
        self.emergency_backoff_time = rospy.get_param("~emergency_backoff_time", 0.8)
        self.backoff_omega_base = rospy.get_param("~backoff_omega", 0.8)

        self.backoff_active = False
        self.backoff_end_time = None
        self.backoff_omega = 0.0

        # 상태
        self.current_scan = None
        self.stage = "approach"
        self.pass_complete_start_time = None
        self.is_passed = False

        self.trigger_reset_done = False
        self.auto_trigger_enabled = rospy.get_param("~auto_trigger_enabled", False)
        self.is_active = not self.auto_trigger_enabled

        # 경로 / 갭 잠금
        self.locked_path_name = None
        self.locked_gap_center_deg = None
        self.locked_gap_width = None

        # ROS pub/sub
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.status_pub = rospy.Publisher("/m3_gap_follower/status", String, queue_size=1)
        self.pass_complete_pub = rospy.Publisher("/m3_gap_follower/pass_complete", String, queue_size=1, latch=True)

        if self.auto_trigger_enabled:
            rospy.Subscriber("/m3_auto_trigger/trigger", Bool, self.trigger_callback, queue_size=1)

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("[M3GapFollower] ===== Initialized =====")
        rospy.loginfo("[M3GapFollower] Topics: scan=%s, cmd=%s", self.scan_topic, self.cmd_topic)
        rospy.loginfo("[M3GapFollower] Robot width=%.2fm, Safety margin=%.2fm", self.robot_width, self.side_safety_margin)
        rospy.loginfo("[M3GapFollower] Speed: nominal=%.2f, min=%.2f, omega_max=%.2f", 
                     self.v_nominal, self.v_min, self.omega_max)
        rospy.loginfo("[M3GapFollower] Init OK (auto_trigger_enabled=%s, debug=%s, stage=%s)", 
                     self.auto_trigger_enabled, self.debug, self.stage)
    
    def stop_robot(self):
        """로봇 정지"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    def scan_callback(self, msg):
        self.current_scan = msg
        if self.debug:
            rospy.logdebug_throttle(2.0, "[M3GapFollower] Scan: %d points, range=[%.2f, %.2f]", 
                                   len(msg.ranges), msg.range_min, msg.range_max)

    def trigger_callback(self, msg):
        if msg.data:
            self.is_active = True
            rospy.logwarn("[M3GapFollower] ✅ 자동 트리거 ON, M3 모드 시작")
        else:
            self.is_active = False
            rospy.loginfo("[M3GapFollower] 자동 트리거 OFF")

    def lidar_to_robot_frame(self, angle_l_deg, _range_val):
        return angle_l_deg

    # ------------------------------------------------------------------
    def calculate_open_space_ratio(self, points, angle_min, angle_max):
        zone_points = [p for p in points if angle_min <= p['heading_deg'] <= angle_max]
        if not zone_points:
            return 0.0
        open_count = sum(1 for p in zone_points if self.open_space_range_min <= p['range'] <= self.open_space_range_max)
        return (open_count / float(len(zone_points))) * 100.0

    def find_gaps_in_path(self, points, angle_min, angle_max, path_name="unknown"):
        if angle_min > angle_max:
            angle_min, angle_max = angle_max, angle_min

        zone_points = [p for p in points if angle_min <= p['heading_deg'] <= angle_max]
        if self.debug:
            rospy.loginfo_throttle(1.0, "[M3GapFollower] [DEBUG] %s 갭 탐색: %.1f~%.1f deg, pts=%d",
                                   path_name.upper(), angle_min, angle_max, len(zone_points))

        if not zone_points:
            return []

        open_points = [p for p in zone_points if self.gap_detect_range_min <= p['range'] <= self.gap_detect_range_max]
        if not open_points:
            if self.debug:
                rospy.logwarn_throttle(1.0, "[M3GapFollower] [DEBUG] %s 열린 포인트 없음", path_name.upper())
            return []

        open_points.sort(key=lambda p: p['heading_deg'])
        headings = [p['heading_deg'] for p in open_points]

        gaps = []
        gap_start_angle = headings[0]
        prev_angle = headings[0]

        for h in headings[1:]:
            if h - prev_angle > float(self.gap_split_deg):
                gap_end_angle = prev_angle
                gap_w = gap_end_angle - gap_start_angle
                if gap_w >= self.min_gap_width_deg:
                    gaps.append({
                        'start_angle': gap_start_angle,
                        'end_angle': gap_end_angle,
                        'center_angle': 0.5 * (gap_start_angle + gap_end_angle),
                        'width': gap_w
                    })
                gap_start_angle = h
            prev_angle = h

        gap_end_angle = prev_angle
        gap_w = gap_end_angle - gap_start_angle
        if gap_w >= self.min_gap_width_deg:
            gaps.append({
                'start_angle': gap_start_angle,
                'end_angle': gap_end_angle,
                'center_angle': 0.5 * (gap_start_angle + gap_end_angle),
                'width': gap_w
            })

        if self.debug:
            rospy.loginfo_throttle(1.0, "[M3GapFollower] [DEBUG] %s 갭=%d개", path_name.upper(), len(gaps))
        return gaps

    @staticmethod
    def select_best_gap(gaps):
        if not gaps:
            return None
        return max(gaps, key=lambda g: g['width'])

    def get_initial_gap_window(self, path_name):
        if path_name == 'left':
            base_min, base_max = self.left_angle_min, self.left_angle_max
        elif path_name == 'center':
            base_min, base_max = self.center_angle_min, self.center_angle_max
        else:
            base_min, base_max = self.right_angle_min, self.right_angle_max
        return base_min - self.gap_search_margin_deg, base_max + self.gap_search_margin_deg

    # ------------------------------------------------------------------
    def compute_corridor_sides(self, points, selected_path_name):
        if selected_path_name == 'left':
            left_sector = (30.0, 90.0)
            right_sector = (-10.0, 30.0)
        elif selected_path_name == 'center':
            left_sector = (10.0, 90.0)
            right_sector = (-90.0, -10.0)
        else:
            left_sector = (-30.0, 10.0)
            right_sector = (-90.0, -30.0)

        left_ranges = [p['range'] for p in points if left_sector[0] <= p['heading_deg'] <= left_sector[1]]
        right_ranges = [p['range'] for p in points if right_sector[0] <= p['heading_deg'] <= right_sector[1]]

        min_left = min(left_ranges) if left_ranges else 999.0
        min_right = min(right_ranges) if right_ranges else 999.0
        return min_left, min_right

    # ------------------------------------------------------------------
    def control_loop(self, _event):
        if self.current_scan is None:
            return

        if self.stage == "done" or self.is_passed:
            self.cmd_pub.publish(Twist())
            return

        if self.auto_trigger_enabled and not self.is_active:
            return

        msg = self.current_scan
        n = len(msg.ranges)
        if n < 10:
            return

        angles_deg = [math.degrees(msg.angle_min + i * msg.angle_increment) for i in range(n)]

        points = []
        for i, r in enumerate(msg.ranges):
            if (not math.isfinite(r)) or (r < msg.range_min) or (r > msg.range_max) or (r == 0.0):
                continue
            heading_deg = self.lidar_to_robot_frame(angles_deg[i], r)
            points.append({'heading_deg': heading_deg, 'range': r})

        if not points:
            if self.debug:
                rospy.logwarn_throttle(2.0, "[M3GapFollower] No valid points in scan")
            self.cmd_pub.publish(Twist())
            return

        headings_all = [p['heading_deg'] for p in points]
        heading_min_all = min(headings_all)
        heading_max_all = max(headings_all)

        # 거리 정보
        front_points = [p for p in points if -30.0 <= p['heading_deg'] <= 30.0]
        min_front = min([p['range'] for p in front_points]) if front_points else 999.0

        center_points = [p for p in points if -15.0 <= p['heading_deg'] <= 15.0]
        min_center = min([p['range'] for p in center_points]) if center_points else 999.0

        side_left_points = [p for p in points if 30.0 <= p['heading_deg'] <= 90.0]
        side_right_points = [p for p in points if -90.0 <= p['heading_deg'] <= -30.0]
        min_side_left = min([p['range'] for p in side_left_points]) if side_left_points else 999.0
        min_side_right = min([p['range'] for p in side_right_points]) if side_right_points else 999.0

        # 백오프 모드
        if self.backoff_active:
            if rospy.Time.now() < self.backoff_end_time:
                cmd = Twist()
                cmd.linear.x = -self.emergency_backoff_speed
                cmd.angular.z = self.backoff_omega
                self.cmd_pub.publish(cmd)
                self.status_pub.publish("backoff")
                if self.debug:
                    rospy.logwarn_throttle(0.5, "[M3GapFollower] ⚠️ BACKOFF active: v=%.2f, w=%.2f", 
                                         cmd.linear.x, cmd.angular.z)
                return
            else:
                rospy.logwarn("[M3GapFollower] Backoff completed")
                self.backoff_active = False
                self.backoff_end_time = None

        # (A) 경로 선택/잠금
        left_ratio = self.calculate_open_space_ratio(points, self.left_angle_min, self.left_angle_max)
        center_ratio = self.calculate_open_space_ratio(points, self.center_angle_min, self.center_angle_max)
        right_ratio = self.calculate_open_space_ratio(points, self.right_angle_min, self.right_angle_max)

        path_ratios = {'left': left_ratio, 'center': center_ratio, 'right': right_ratio}

        if self.locked_path_name is None:
            selected_path_name, selected_ratio = max(path_ratios.items(), key=lambda x: x[1])
            self.locked_path_name = selected_path_name
            rospy.logwarn("[M3GapFollower] ⭐ 경로 잠금: %s (%.1f%%)", selected_path_name.upper(), selected_ratio)
        else:
            selected_path_name = self.locked_path_name
            selected_ratio = path_ratios.get(selected_path_name, 0.0)

        # (B) 갭 탐색 범위
        if self.locked_gap_center_deg is None:
            gap_min, gap_max = self.get_initial_gap_window(selected_path_name)
            gap_min = max(gap_min, heading_min_all)
            gap_max = min(gap_max, heading_max_all)
        else:
            half = self.gap_track_window_deg
            gap_min = max(self.locked_gap_center_deg - half, heading_min_all)
            gap_max = min(self.locked_gap_center_deg + half, heading_max_all)
            if gap_min > gap_max:
                gap_min, gap_max = gap_max, gap_min

        # (C) 갭 탐색 & 최초 1회 잠금
        best_gap = None
        if self.stage != "corridor":
            gaps = self.find_gaps_in_path(points, gap_min, gap_max, selected_path_name)
            best_gap = self.select_best_gap(gaps)

            if self.locked_gap_center_deg is None:
                if not best_gap:
                    self.cmd_pub.publish(Twist())
                    self.status_pub.publish(f"path={selected_path_name},ratio={selected_ratio:.1f}%,gap=none,stage={self.stage}")
                    return

                candidate_center = best_gap['center_angle']
                if selected_path_name == 'right':
                    candidate_center -= self.outward_angle_bias_deg
                elif selected_path_name == 'left':
                    candidate_center += self.outward_angle_bias_deg

                candidate_center = np.clip(
                    candidate_center,
                    best_gap['start_angle'] + self.gap_margin_deg,
                    best_gap['end_angle'] - self.gap_margin_deg
                )

                self.locked_gap_center_deg = float(candidate_center)
                self.locked_gap_width = float(best_gap['width'])

                rospy.logwarn("[M3GapFollower] ⭐ 갭 중심 잠금: %.1f° (width=%.1f°, path=%s)",
                              self.locked_gap_center_deg, self.locked_gap_width, selected_path_name.upper())

        # (D) 기본 조향각 계산
        if self.stage == "corridor":
            gap_center_deg = 0.0
            gap_width = self.locked_gap_width if self.locked_gap_width is not None else 40.0
        else:
            if self.locked_gap_center_deg is None:
                self.cmd_pub.publish(Twist())
                return
            raw_center = self.locked_gap_center_deg
            gap_center_deg = float(np.clip(raw_center, -self.max_approach_angle_deg, self.max_approach_angle_deg))
            gap_width = self.locked_gap_width if self.locked_gap_width is not None else (best_gap['width'] if best_gap else 40.0)

        omega = self.omega_gain * math.radians(gap_center_deg)

        # (F) 통로 경계 계산
        corr_left, corr_right = self.compute_corridor_sides(points, selected_path_name)

        # (G) stage 전환
        if self.stage == "approach":
            in_left = (self.corridor_side_min <= corr_left <= self.corridor_side_max)
            in_right = (self.corridor_side_min <= corr_right <= self.corridor_side_max)
            side_diff = abs(corr_left - corr_right)
            front_ok = (min_front <= self.corridor_front_max)

            if in_left and in_right and side_diff <= self.corridor_side_diff_max and front_ok:
                self.stage = "corridor"
                rospy.logwarn("[M3GapFollower] 🚪 통로 진입 -> CORRIDOR (L=%.2f R=%.2f front=%.2f)",
                              corr_left, corr_right, min_front)
                self.pass_complete_start_time = None

        elif self.stage == "corridor":
            front_clear = (min_front >= self.pass_front_clear_dist)
            side_cleared = (
                (corr_left - self.corridor_side_max) >= self.corridor_clear_margin or
                (corr_right - self.corridor_side_max) >= self.corridor_clear_margin
            )
            if front_clear and side_cleared:
                if self.pass_complete_start_time is None:
                    self.pass_complete_start_time = rospy.Time.now()
                    rospy.loginfo("[M3GapFollower] Corridor cleared, starting pass complete timer...")
                else:
                    elapsed = (rospy.Time.now() - self.pass_complete_start_time).to_sec()
                    if elapsed >= self.pass_complete_time and not self.is_passed:
                        self.is_passed = True
                        self.stage = "done"
                        info = f"completed,path={selected_path_name},L={corr_left:.2f},R={corr_right:.2f},front={min_front:.2f}"
                        rospy.logwarn("[M3GapFollower] ✅ 통로 통과 완료! (%s, elapsed=%.2fs)", 
                                     selected_path_name.upper(), elapsed)
                        self.pass_complete_pub.publish(info)

                        # 트리거 리셋
                        if self.auto_trigger_enabled and not self.trigger_reset_done:
                            try:
                                rospy.wait_for_service("/m3_auto_trigger/set_trigger", timeout=0.5)
                                reset_srv = rospy.ServiceProxy("/m3_auto_trigger/set_trigger", SetBool)
                                reset_srv(False)
                            except Exception as e:
                                rospy.logwarn("[M3GapFollower] 트리거 리셋 실패: %s", e)
                            self.trigger_reset_done = True
                            self.is_active = False
            else:
                self.pass_complete_start_time = None

        # (H) 벽/장애물 기반 조향 보정 + 백오프 트리거
        if self.wall_follow_enabled:
            if (self.emergency_backoff_enabled and not self.backoff_active and
                    (min_side_left < self.emergency_backoff_dist or min_side_right < self.emergency_backoff_dist)):

                if min_side_right <= min_side_left:
                    dir_sign = 1.0   # 왼쪽 회전
                else:
                    dir_sign = -1.0  # 오른쪽 회전

                self.backoff_active = True
                self.backoff_end_time = rospy.Time.now() + rospy.Duration(self.emergency_backoff_time)
                self.backoff_omega = float(dir_sign * self.backoff_omega_base)

                cmd = Twist()
                cmd.linear.x = -self.emergency_backoff_speed
                cmd.angular.z = self.backoff_omega
                self.cmd_pub.publish(cmd)
                self.status_pub.publish("backoff")
                return

            omega_correction = 0.0

            if min_side_left < self.wall_critical_dist:
                omega_correction = -3.0 * self.center_balance_gain * (self.wall_critical_dist - min_side_left)
            elif min_side_right < self.wall_critical_dist:
                omega_correction = 3.0 * self.center_balance_gain * (self.wall_critical_dist - min_side_right)
            else:
                if min_center < self.center_avoid_dist:
                    if corr_left > corr_right:
                        dir_sign = 1.0
                    elif corr_right > corr_left:
                        dir_sign = -1.0
                    else:
                        dir_sign = 0.0
                    if dir_sign != 0.0:
                        omega_correction = dir_sign * self.center_avoid_gain * (self.center_avoid_dist - min_center)
                else:
                    if self.stage == "corridor":
                        if corr_left < 3.0 * self.corridor_width and corr_right < 3.0 * self.corridor_width:
                            dist_diff = corr_left - corr_right
                            omega_correction = (self.center_balance_gain * 1.5) * dist_diff
                    else:
                        if (corr_left < self.pre_corridor_center_dist and corr_right < self.pre_corridor_center_dist and
                                corr_left < 3.0 * self.corridor_width and corr_right < 3.0 * self.corridor_width):
                            dist_diff = corr_left - corr_right
                            omega_correction = self.pre_center_balance_gain * dist_diff

            omega += omega_correction

        omega = float(np.clip(omega, -self.omega_max, self.omega_max))

        # (I) 속도 계산
        v = float(self.v_nominal)

        if gap_width < 20.0:
            v *= 0.5
        elif gap_width < 30.0:
            v *= 0.65
        elif gap_width < 40.0:
            v *= 0.8

        if abs(omega) > 1.2:
            v *= 0.55
        elif abs(omega) > 0.8:
            v *= 0.7

        if min_center < 0.30:
            v *= 0.5
        elif min_center < 0.50:
            v *= 0.65
        elif min_center < 0.70:
            v *= 0.8
        elif min_front < 1.0:
            v *= 0.85

        v = max(self.v_min, v)
        if abs(gap_center_deg) < 5.0 and v < self.v_min * 1.5:
            v = self.v_min * 1.5

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        self.status_pub.publish(
            f"path={selected_path_name},ratio={selected_ratio:.1f}%,gap_center={gap_center_deg:.1f}°,width={gap_width:.1f}°,stage={self.stage}"
        )


if __name__ == "__main__":
    node = None
    try:
        node = M3GapFollower()
        
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