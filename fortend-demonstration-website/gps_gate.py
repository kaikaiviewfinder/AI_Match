#!/usr/bin/env python3
"""GPS gate: throttle continuous /gps_raw to a sparse, IRREGULAR /gps.

Simulates weak / intermittent GPS: after each published fix, the next fix is
released only after the receiver has moved a randomly varying distance
(mostly 4-18 m, occasionally a longer 20-35 m dropout). The very first fix is
always published so the route matcher still gets an initial anchor.
"""
import math
import random
import rospy
from sensor_msgs.msg import NavSatFix


def haversine(lat1, lon1, lat2, lon2):
    R = 6378137.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


class GpsGate:
    def __init__(self):
        self.near_min = float(rospy.get_param("~near_min_m", 4.0))
        self.near_max = float(rospy.get_param("~near_max_m", 18.0))
        self.drop_min = float(rospy.get_param("~drop_min_m", 20.0))
        self.drop_max = float(rospy.get_param("~drop_max_m", 35.0))
        self.drop_prob = float(rospy.get_param("~drop_prob", 0.15))
        self.last_pub = None
        self.next_dist = self._draw()
        self.n_in = 0
        self.n_out = 0
        self.pub = rospy.Publisher("/gps", NavSatFix, queue_size=10)
        rospy.Subscriber("/gps_raw", NavSatFix, self.cb, queue_size=100)
        rospy.loginfo("gps_gate: /gps_raw -> /gps, irregular %.0f-%.0fm (%.0f%% dropouts %.0f-%.0fm)",
                      self.near_min, self.near_max, self.drop_prob * 100, self.drop_min, self.drop_max)

    def _draw(self):
        if random.random() < self.drop_prob:
            return random.uniform(self.drop_min, self.drop_max)
        return random.uniform(self.near_min, self.near_max)

    def cb(self, msg):
        self.n_in += 1
        if self.last_pub is None:
            self.last_pub = (msg.latitude, msg.longitude)
            self.n_out += 1
            self.pub.publish(msg)
            return
        d = haversine(self.last_pub[0], self.last_pub[1], msg.latitude, msg.longitude)
        if d >= self.next_dist:
            self.last_pub = (msg.latitude, msg.longitude)
            self.next_dist = self._draw()
            self.n_out += 1
            self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("gps_gate")
    GpsGate()
    rospy.spin()
