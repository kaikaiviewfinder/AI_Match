#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read BME280 (temperature + humidity + pressure) from STM32 serial,
convert pressure to altitude, publish /baro_altitude (Float64).

Expected STM32 serial format:
  Temperature: 25.37 C  |  Humidity: 65.00 %  |  Pressure: 100123.00 Pa

Altitude uses ISA standard atmosphere model.
Set sea_level_pressure to a local QNH reference for best accuracy.

Usage:
  rosrun hiking_route_localization bme280_altitude_node.py _port:=/dev/ttyUSB0
  rosrun hiking_route_localization bme280_altitude_node.py _simulate:=true
"""

import math
import re

import rospy
from std_msgs.msg import Float64, String


def pressure_to_altitude(pressure_hpa, sea_level_hpa=1013.25, temperature_c=15.0):
    """ISA standard atmosphere: pressure (hPa) → altitude (m)."""
    t0 = temperature_c + 273.15
    return (t0 / 0.0065) * (1.0 - (pressure_hpa / sea_level_hpa) ** 0.190263)


class BME280AltitudeNode:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyUSB0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.sea_level_hpa = float(rospy.get_param("~sea_level_hpa", 1013.25))
        self.simulate = bool(rospy.get_param("~simulate", False))

        # Simulation parameters
        self.sim_base_pres = float(rospy.get_param("~sim_base_pres", 1013.25))
        self.sim_rate = float(rospy.get_param("~sim_rate", 20.0))

        self.pattern = re.compile(
            r"Temperature:\s*(?P<temp>[\d.]+)\s*C\s*\|\s*"
            r"Humidity:\s*(?P<hum>[\d.]+)\s*%\s*\|\s*"
            r"Pressure:\s*(?P<pres>[\d.]+)\s*Pa"
        )

        self.pub = rospy.Publisher("/baro_altitude", Float64, queue_size=50)
        self.raw_pub = rospy.Publisher("/baro_altitude/raw_line", String, queue_size=10)
        self.temp_pub = rospy.Publisher("/baro_altitude/temperature", Float64, queue_size=10)
        self.pres_pub = rospy.Publisher("/baro_altitude/pressure", Float64, queue_size=10)

    def parse_line(self, line: str):
        """Parse BME280 line. Returns (temp_C, humidity_pct, pressure_hPa, altitude_m) or None."""
        match = self.pattern.search(line)
        if not match:
            return None
        temp = float(match.group("temp"))
        hum = float(match.group("hum"))
        pres_pa = float(match.group("pres"))
        pres_hpa = pres_pa / 100.0
        alt = pressure_to_altitude(pres_hpa, self.sea_level_hpa, temp)
        return temp, hum, pres_hpa, alt

    def run_sim(self):
        import random
        rate = rospy.Rate(self.sim_rate)
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - t0
            # Simulate climbing with slow pressure drop
            pres_hpa = self.sim_base_pres - 0.012 * t + random.gauss(0.0, 0.1)
            alt = pressure_to_altitude(pres_hpa, self.sim_base_pres, 15.0)
            self.pub.publish(Float64(data=alt))
            rate.sleep()

    def run_serial(self):
        try:
            import serial
        except ImportError:
            rospy.logerr("pyserial not installed. Run: sudo apt install python3-serial")
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=2.0)
        except Exception as exc:
            rospy.logerr("Cannot open %s: %s", self.port, exc)
            return
        rospy.loginfo("Reading BME280 from %s @ %d baud, sea-level ref %.2f hPa",
                      self.port, self.baud, self.sea_level_hpa)

        while not rospy.is_shutdown():
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "Serial read error: %s", exc)
                continue
            if not line:
                continue
            self.raw_pub.publish(String(data=line))
            result = self.parse_line(line)
            if result is None:
                continue
            temp, hum, pres_hpa, alt = result
            self.pub.publish(Float64(data=alt))
            self.temp_pub.publish(Float64(data=temp))
            self.pres_pub.publish(Float64(data=pres_hpa))


def main():
    rospy.init_node("bme280_altitude_node")
    node = BME280AltitudeNode()
    if node.simulate:
        node.run_sim()
    else:
        node.run_serial()


if __name__ == "__main__":
    main()
