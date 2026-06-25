#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read STM32 barometric altitude from serial and publish /baro_altitude.
Expected line formats:
  timestamp_ms,pressure_pa,temperature_c,altitude_m
  pressure_pa,temperature_c,altitude_m
  altitude_m
Examples:
  1719040123456,100436.37,23.79,421.82
  100436.37,23.79,421.82
  421.82
"""
import math
import re

import rospy
from std_msgs.msg import Float64, String


class SerialAltitudeNode:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyUSB0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.simulate = bool(rospy.get_param("~simulate", False))
        self.sim_base_alt = float(rospy.get_param("~sim_base_alt", 120.0))
        self.sim_rate = float(rospy.get_param("~sim_rate", 20.0))
        self.pub = rospy.Publisher("/baro_altitude", Float64, queue_size=50)
        self.raw_pub = rospy.Publisher("/baro_altitude/raw_line", String, queue_size=10)

    def parse_altitude(self, line: str):
        line = line.strip()
        if not line:
            return None
        # Extract all floating numbers; use the last one as altitude.
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except ValueError:
            return None

    def run_sim(self):
        rate = rospy.Rate(self.sim_rate)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - t0
            z = self.sim_base_alt + 0.08 * t + 2.0 * math.sin(t / 9.0)
            self.pub.publish(Float64(data=z))
            rate.sleep()

    def run_serial(self):
        try:
            import serial
        except ImportError:
            rospy.logerr("pyserial is not installed. Install it: sudo apt install python3-serial, or run with _simulate:=true")
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except Exception as exc:
            rospy.logerr("failed to open serial port %s: %s", self.port, exc)
            return
        rospy.loginfo("reading altitude from %s @ %d", self.port, self.baud)
        while not rospy.is_shutdown():
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "serial read error: %s", exc)
                continue
            if not raw:
                continue
            self.raw_pub.publish(String(data=raw))
            alt = self.parse_altitude(raw)
            if alt is None:
                rospy.logwarn_throttle(2.0, "cannot parse altitude line: %s", raw)
                continue
            self.pub.publish(Float64(data=alt))


def main():
    rospy.init_node("serial_altitude_node")
    node = SerialAltitudeNode()
    if node.simulate:
        node.run_sim()
    else:
        node.run_serial()


if __name__ == "__main__":
    main()
