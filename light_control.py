#! /usr/bin/env python

import rospy
from scout_msgs.msg import ScoutLightCmd

class Light_Control():
    def __init__(self):
        rospy.init_node('scout_light_control_shate', anonymous=False)
        self.msg_pub = rospy.Publisher(
            '/scout_light_control',
            ScoutLightCmd,
            queue_size=10
        )
        self.count = 0

    def onMsg(self):
        on = ScoutLightCmd()
        on.enable_cmd_light_control = True
        on.front_mode = 1
        self.msg_pub.publish(on)

    def offMsg(self):
        off = ScoutLightCmd()
        off.enable_cmd_light_control = True
        off.front_mode = 0
        self.msg_pub.publish(off)

# 1초 간격으로 깜빡임
def blink():
    count = 0
    while not rospy.is_shutdown():
        light = Light_Control()
        rate = rospy.Rate(10)
        if str(count)[-1] == '1': light.onMsg() # 1초 주기로 on
        else: light.offMsg() # 나머지는 off
        rate.sleep()
        count+=1

# 조명 켜기
def on():
    while not rospy.is_shutdown():
        light = Light_Control()
        rate = rospy.Rate(10)
        light.onMsg()
        rate.sleep()

# 조명 끄기
def off():
    while not rospy.is_shutdown():
        light = Light_Control()
        rate = rospy.Rate(10)
        light.offMsg()
        rate.sleep()

def emergency():
    count = 0
    while not rospy.is_shutdown():
        light = Light_Control()
        rate = rospy.Rate(60)
        if str(count)[-1] in ['1', '6']: light.onMsg() # 1초 주기로 on
        else: light.offMsg() # 나머지는 off
        rate.sleep()
        count+=1

if __name__ == "__main__":
    blink()
    # off()
    # on()
    # emergency()