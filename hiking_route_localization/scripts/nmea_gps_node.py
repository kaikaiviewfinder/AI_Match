#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read NMEA GPS from serial port and publish /gps (sensor_msgs/NavSatFix).

Supports standard NMEA sentences: $GPGGA, $GPRMC, $GPGLL, $GNGGA, $GNRMC.
Publishes NavSatFix with position_covariance when fix is valid.

Usage:
  rosrun hiking_route_localization nmea_gps_node.py _port:=/dev/ttyUSB0 _baud:=9600
  rosrun hiking_route_localization nmea_gps_node.py _simulate:=true
"""

import math
import re
import time

import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus


def nmea_checksum_ok(sentence: str) -> bool:
    """Verify NMEA checksum (e.g. $GPGGA,...*5C)."""
    m = re.match(r"\$(.+)\*([0-9A-Fa-f]{2})$", sentence)
    if not m:
        return False
    body, csum = m.group(1), m.group(2)
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return cs == int(csum, 16)


def parse_lat_lon(fields, lat_idx, lat_dir_idx, lon_idx, lon_dir_idx):
    """Parse ddmm.mmmm NMEA lat/lon to decimal degrees. Returns (lat, lon) or None."""
    try:
        lat_raw = float(fields[lat_idx])
        lat_dir = fields[lat_dir_idx]
        lon_raw = float(fields[lon_idx])
        lon_dir = fields[lon_dir_idx]
    except (ValueError, IndexError):
        return None

    lat_deg = int(lat_raw / 100.0)
    lat_min = lat_raw - lat_deg * 100.0
    lat = lat_deg + lat_min / 60.0
    if lat_dir == "S":
        lat = -lat

    lon_deg = int(lon_raw / 100.0)
    lon_min = lon_raw - lon_deg * 100.0
    lon = lon_deg + lon_min / 60.0
    if lon_dir == "W":
        lon = -lon

    return lat, lon


class NMEAGPSNode:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyUSB0")
        self.baud = int(rospy.get_param("~baud", 9600))
        self.frame_id = rospy.get_param("~frame_id", "gps")
        self.simulate = bool(rospy.get_param("~simulate", False))

        # Simulate parameters
        self.sim_lat = float(rospy.get_param("~sim_lat", 22.53))
        self.sim_lon = float(rospy.get_param("~sim_lon", 113.9532))
        self.sim_alt = float(rospy.get_param("~sim_alt", 120.0))
        self.sim_rate = float(rospy.get_param("~sim_rate", 1.0))
        self.sim_noise = float(rospy.get_param("~sim_noise", 5.0))
        self.sim_dropout = float(rospy.get_param("~sim_dropout", 0.4))

        self.pub = rospy.Publisher("/gps", NavSatFix, queue_size=20)

    def publish_fix(self, lat, lon, alt, covariance=None):
        msg = NavSatFix()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        if covariance is not None:
            msg.position_covariance = [
                covariance, 0.0, 0.0,
                0.0, covariance, 0.0,
                0.0, 0.0, covariance * 4.0,
            ]
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        else:
            msg.position_covariance = [0.0] * 9
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.pub.publish(msg)

    def run_sim(self):
        import random
        rate = rospy.Rate(self.sim_rate)
        while not rospy.is_shutdown():
            if random.random() > self.sim_dropout:
                lat = self.sim_lat + random.gauss(0.0, self.sim_noise / 111320.0)
                lon = self.sim_lon + random.gauss(0.0, self.sim_noise / (111320.0 * math.cos(math.radians(self.sim_lat))))
                alt = self.sim_alt + random.gauss(0.0, 8.0)
                self.publish_fix(lat, lon, alt, self.sim_noise ** 2)
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
        rospy.loginfo("Reading NMEA GPS from %s @ %d baud", self.port, self.baud)

        latest_lat = latest_lon = latest_alt = None
        last_pub = 0.0
        while not rospy.is_shutdown():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "Serial read error: %s", exc)
                continue
            if not line or not line.startswith("$"):
                continue
            if not nmea_checksum_ok(line):
                continue

            fields = line.split(",")
            talker = fields[0]

            # $--GGA: Global Positioning System Fix Data
            if talker.endswith("GGA") and len(fields) >= 10:
                result = parse_lat_lon(fields, 2, 3, 4, 5)
                if result is None:
                    continue
                latest_lat, latest_lon = result
                try:
                    latest_alt = float(fields[9])
                except (ValueError, IndexError):
                    latest_alt = 0.0
                try:
                    quality = int(fields[6])
                except (ValueError, IndexError):
                    quality = 0

                if quality > 0 and latest_lat is not None:
                    self.publish_fix(latest_lat, latest_lon, latest_alt)
                    last_pub = time.time()

            # $--RMC: Recommended Minimum Navigation Information
            elif talker.endswith("RMC") and len(fields) >= 6:
                if fields[2] != "A":  # Status: A=active, V=void
                    continue
                result = parse_lat_lon(fields, 3, 4, 5, 6)
                if result is not None:
                    latest_lat, latest_lon = result

            # Periodically re-publish even without new GGA
            if latest_lat is not None and time.time() - last_pub > 5.0:
                self.publish_fix(latest_lat, latest_lon, latest_alt or 0.0)
                last_pub = time.time()


def main():
    rospy.init_node("nmea_gps_node")
    node = NMEAGPSNode()
    if node.simulate:
        node.run_sim()
    else:
        node.run_serial()


if __name__ == "__main__":
    main()
