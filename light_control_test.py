#! /usr/bin/env python

from light_control import *
from scout_msgs.msg import ScoutLightCmd
import rospy

if __name__ == '__main__':
    blink() # 실행해보면 실제로 로봇의 조명이 깜빡인다