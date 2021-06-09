#! /usr/bin/env python

from light_control import *
from scout_msgs.msg import ScoutLightCmd
import rospy

if __name__ == '__main__':
    blink() # 천천히 깜빡거리기(Target Lost 상태)
    # on() # 조명 켜기(주행 준비 완료, 주행 중:이상 무)
    # off() # 조명 끄기(주행 준비 전 상태)
    # emergency() # 빠르게 깜빡거리기 (정지 또는 좌/우 이동)