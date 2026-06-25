#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo publisher: fake VINS odometry, intermittent GPS and barometer from a route csv."""
import math
import os
import random
import sys

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from route_tools import RouteTable, local_xy_to_lla


class DemoSensorPlayer:
    def __init__(self):
        default_route = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "demo_route.csv")
        self.route_csv = rospy.get_param("~route_csv", default_route)
        self.route = RouteTable.load_csv(self.route_csv)
        self.speed_mps = float(rospy.get_param("~speed_mps", 1.2))
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.gps_rate_hz = float(rospy.get_param("~gps_rate_hz", 1.0))
        self.gps_noise_m = float(rospy.get_param("~gps_noise_m", 6.0))
        self.gps_dropout_ratio = float(rospy.get_param("~gps_dropout_ratio", 0.45))
        self.baro_noise_m = float(rospy.get_param("~baro_noise_m", 1.2))
        self.vio_step_scale = float(rospy.get_param("~vio_step_scale", 1.01))
        self.loop = bool(rospy.get_param("~loop", True))
        self.start_s = float(rospy.get_param("~start_s", 0.0))

        self.odom_pub = rospy.Publisher("/vins_estimator/odometry", Odometry, queue_size=50)
        self.gps_pub = rospy.Publisher("/gps", NavSatFix, queue_size=10)
        self.baro_pub = rospy.Publisher("/baro_altitude", Float64, queue_size=50)

        self.s_true = self.start_s
        self.vio_x = 0.0
        self.vio_y = 0.0
        self.vio_z = 0.0
        self.last_route_point = self.route.interpolate(self.s_true)
        self.last_gps_t = 0.0
        self.t0 = rospy.Time.now().to_sec()
        rospy.loginfo("demo_sensor_player route length %.1f m", self.route.length)

    def publish_odom(self, stamp, dt):
        prev = self.last_route_point
        self.s_true += self.speed_mps * dt
        if self.s_true > self.route.length:
            if self.loop:
                self.s_true = 0.0
                self.vio_x = self.vio_y = self.vio_z = 0.0
            else:
                self.s_true = self.route.length
        cur = self.route.interpolate(self.s_true)
        ds = math.sqrt((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2 + (cur.z - prev.z) ** 2)
        # Simulate VIO as local arbitrary path. We only need its incremental distance for matcher.
        yaw = cur.yaw + 0.05 * math.sin((stamp.to_sec() - self.t0) / 15.0)
        step = ds * self.vio_step_scale + random.gauss(0.0, 0.02)
        self.vio_x += step * math.cos(yaw)
        self.vio_y += step * math.sin(yaw)
        self.vio_z += (cur.z - prev.z) * self.vio_step_scale + random.gauss(0.0, 0.01)
        self.last_route_point = cur

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "vio_world"
        odom.child_frame_id = "vio_body"
        odom.pose.pose.position.x = self.vio_x
        odom.pose.pose.position.y = self.vio_y
        odom.pose.pose.position.z = self.vio_z
        half = 0.5 * yaw
        odom.pose.pose.orientation.w = math.cos(half)
        odom.pose.pose.orientation.z = math.sin(half)
        odom.twist.twist.linear.x = self.speed_mps
        self.odom_pub.publish(odom)

    def publish_baro(self):
        cur = self.route.interpolate(self.s_true)
        z = cur.z + random.gauss(0.0, self.baro_noise_m)
        self.baro_pub.publish(Float64(data=z))

    def publish_gps_if_due(self, stamp):
        t = stamp.to_sec()
        if t - self.last_gps_t < 1.0 / max(self.gps_rate_hz, 1e-3):
            return
        self.last_gps_t = t
        if random.random() < self.gps_dropout_ratio:
            # Publish no message to simulate dropout.
            return
        cur = self.route.interpolate(self.s_true)
        x = cur.x + random.gauss(0.0, self.gps_noise_m)
        y = cur.y + random.gauss(0.0, self.gps_noise_m)
        lon, lat = local_xy_to_lla(x, y, self.route.lon0, self.route.lat0)

        msg = NavSatFix()
        msg.header.stamp = stamp
        msg.header.frame_id = "gps"
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        msg.longitude = lon
        msg.latitude = lat
        msg.altitude = cur.z + random.gauss(0.0, 8.0)
        var = self.gps_noise_m * self.gps_noise_m
        msg.position_covariance = [var, 0, 0, 0, var, 0, 0, 0, 64.0]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        self.gps_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        last_t = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = now.to_sec()
            dt = max(0.0, min(t - last_t, 0.2))
            last_t = t
            self.publish_odom(now, dt)
            self.publish_baro()
            self.publish_gps_if_due(now)
            rate.sleep()


def main():
    rospy.init_node("demo_sensor_player")
    DemoSensorPlayer().run()


if __name__ == "__main__":
    main()
