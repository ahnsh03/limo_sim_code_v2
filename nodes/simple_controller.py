#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 LIMO 컨트롤러 노드
/cmd_vel 토픽으로 속도 명령을 발행합니다.
"""

import rospy
from geometry_msgs.msg import Twist

class SimpleController:
    def __init__(self):
        rospy.init_node('simple_controller', anonymous=True)
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Rate
        self.rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("Simple Controller 노드가 시작되었습니다.")
    
    def run(self):
        while not rospy.is_shutdown():
            # 간단한 전진 명령 예제
            cmd = Twist()
            cmd.linear.x = 0.5  # 0.5 m/s 전진
            cmd.angular.z = 0.0  # 직진
            
            self.cmd_vel_pub.publish(cmd)
            rospy.loginfo("명령 발행: linear.x=%.2f, angular.z=%.2f", 
                         cmd.linear.x, cmd.angular.z)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = SimpleController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

