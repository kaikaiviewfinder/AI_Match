#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RViz markers for known route and matched current position."""
import json
import math
import os
import sys

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from route_tools import RouteTable


class RouteVisualizer:
    def __init__(self):
        default_route = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "demo_route.csv")
        self.route_csv = rospy.get_param("~route_csv", default_route)
        self.route = RouteTable.load_csv(self.route_csv)
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.route_stride = int(rospy.get_param("~route_stride", 2))
        self.marker_pub = rospy.Publisher("/route_matcher/markers", MarkerArray, queue_size=1, latch=True)
        self.current_state = None
        self.start_s = None  # first observed s — marks where THIS video/tracking began
        rospy.Subscriber("/route_matcher/state", String, self.state_callback, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        rospy.loginfo("route_visualizer loaded route: %s", self.route_csv)

    def state_callback(self, msg: String):
        try:
            self.current_state = json.loads(msg.data)
            # Only capture start_s after GPS has anchored the particle filter.
            # Before GPS, the matcher publishes s≈0 (default init), which is wrong
            # for mid-segment recordings.  We wait until gps_age_s is non-null,
            # meaning at least one GPS fix has been processed.
            if self.start_s is None and self.current_state is not None:
                gps_age = self.current_state.get("gps_age_s")
                if gps_age is not None:
                    self.start_s = float(self.current_state.get("s_m", 0.0))
                    rospy.loginfo("start_s captured after GPS fix: %.1f m", self.start_s)
        except Exception:
            self.current_state = None

    def base_marker(self, marker_id, marker_type, ns="route"):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def timer_callback(self, _event):
        arr = MarkerArray()

        line = self.base_marker(0, Marker.LINE_STRIP, ns="known_route")
        line.scale.x = 1.2
        line.color.r = 0.55
        line.color.g = 0.55
        line.color.b = 0.55
        line.color.a = 1.0
        for i, p in enumerate(self.route.points):
            if i % max(self.route_stride, 1) != 0 and i != len(self.route.points) - 1:
                continue
            q = Point(x=p.x, y=p.y, z=p.z)
            line.points.append(q)
        arr.markers.append(line)

        if self.current_state is not None:
            x = float(self.current_state.get("matched_x", 0.0))
            y = float(self.current_state.get("matched_y", 0.0))
            z = float(self.current_state.get("matched_z", 0.0))
            s = float(self.current_state.get("s_m", 0.0))
            progress = 100.0 * float(self.current_state.get("progress", 0.0))
            conf = float(self.current_state.get("confidence", 0.0))
            dist_end = float(self.current_state.get("distance_to_end_m", 0.0))
            seg = int(self.current_state.get("segment_id", 0))

            sphere = self.base_marker(1, Marker.SPHERE, ns="matched_position")
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z + 1.5
            sphere.scale.x = 5.0
            sphere.scale.y = 5.0
            sphere.scale.z = 5.0
            sphere.color.r = 1.0
            sphere.color.g = 0.25
            sphere.color.b = 0.05
            sphere.color.a = 1.0
            arr.markers.append(sphere)

            text = self.base_marker(2, Marker.TEXT_VIEW_FACING, ns="route_text")
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 9.0
            text.scale.z = 5.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"s={s:.1f}m  seg={seg}\nprogress={progress:.1f}%  conf={conf:.2f}\nend={dist_end:.1f}m"
            arr.markers.append(text)

            # Progress line from THIS run's start to current matched s (not from route origin).
            prog_line = self.base_marker(3, Marker.LINE_STRIP, ns="matched_progress")
            prog_line.scale.x = 2.2
            prog_line.color.r = 1.0
            prog_line.color.g = 0.55
            prog_line.color.b = 0.05
            prog_line.color.a = 1.0
            start = self.start_s if self.start_s is not None else 0.0
            for p in self.route.points:
                if start <= p.s <= s:
                    prog_line.points.append(Point(x=p.x, y=p.y, z=p.z + 0.3))
            arr.markers.append(prog_line)

        self.marker_pub.publish(arr)


def main():
    rospy.init_node("route_visualizer_node")
    RouteVisualizer()
    rospy.spin()


if __name__ == "__main__":
    main()
