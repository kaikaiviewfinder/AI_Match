#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle-filter route matcher.
Inputs:
  /vins_estimator/odometry       nav_msgs/Odometry
  /gps                           sensor_msgs/NavSatFix       optional/intermittent
  /baro_altitude                 std_msgs/Float64            optional, meters
Outputs:
  /route_matcher/state           std_msgs/String             JSON state
  /route_matcher/matched_odometry nav_msgs/Odometry           route-constrained pose
  /route_matcher/progress        std_msgs/Float64            0~1 route progress

Design:
- State is one-dimensional route progress s in meters.
- VINS odometry contributes incremental distance.
- GPS updates absolute likelihood against route XY.
- Barometric altitude updates vertical likelihood against route Z.
- Slope likelihood uses altitude trend and route slope.
"""
import json
import math
import os
import random
import sys
from typing import List, Optional, Tuple

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from route_tools import RouteTable, lla_to_local_xy, clamp


class ParticleFilterRouteMatcher:
    def __init__(self):
        default_route = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "demo_route.csv")
        self.route_csv = rospy.get_param("~route_csv", default_route)
        self.route = RouteTable.load_csv(self.route_csv)

        self.num_particles = int(rospy.get_param("~num_particles", 1200))
        self.initial_s = float(rospy.get_param("~initial_s", 0.0))
        self.initial_std_m = float(rospy.get_param("~initial_std_m", 30.0))
        self.uniform_init_without_gps = bool(rospy.get_param("~uniform_init_without_gps", False))

        self.sigma_motion = float(rospy.get_param("~sigma_motion", 1.2))
        self.sigma_gps_default = float(rospy.get_param("~sigma_gps", 8.0))
        self.sigma_alt = float(rospy.get_param("~sigma_alt", 4.0))
        self.sigma_slope = float(rospy.get_param("~sigma_slope", 1.8))
        self.motion_noise = float(rospy.get_param("~motion_noise", 0.4))
        self.gps_gate_m = float(rospy.get_param("~gps_gate_m", 80.0))
        self.resample_neff_ratio = float(rospy.get_param("~resample_neff_ratio", 0.55))
        self.max_step_m = float(rospy.get_param("~max_step_m", 5.0))
        self.allow_backward = bool(rospy.get_param("~allow_backward", False))

        self.weight_gps = float(rospy.get_param("~weight_gps", 1.0))
        self.weight_alt = float(rospy.get_param("~weight_alt", 1.0))
        self.weight_slope = float(rospy.get_param("~weight_slope", 0.5))
        self.weight_motion = float(rospy.get_param("~weight_motion", 0.8))

        # If true, first barometer value is aligned to route altitude at current estimate.
        self.auto_baro_offset = bool(rospy.get_param("~auto_baro_offset", True))
        self.baro_offset = rospy.get_param("~baro_offset", None)
        if self.baro_offset is not None:
            self.baro_offset = float(self.baro_offset)
        self.baro_history_window = float(rospy.get_param("~baro_history_window", 8.0))

        self.particles: List[float] = []
        self.weights: List[float] = []
        self.initialized = False
        self.last_odom_pos: Optional[Tuple[float, float, float]] = None
        self.last_odom_stamp: Optional[float] = None
        self.latest_gps: Optional[Tuple[float, float, float, float]] = None  # x,y,sigma,stamp
        self.latest_baro_raw: Optional[Tuple[float, float]] = None  # z_raw,stamp
        self.baro_trace: List[Tuple[float, float]] = []  # stamp, effective_z
        self.last_s_est: Optional[float] = None
        self.last_state_stamp: Optional[float] = None
        self.last_delta_d: float = 0.0

        self.state_pub = rospy.Publisher("/route_matcher/state", String, queue_size=10)
        self.odom_pub = rospy.Publisher("/route_matcher/matched_odometry", Odometry, queue_size=10)
        self.progress_pub = rospy.Publisher("/route_matcher/progress", Float64, queue_size=10)

        rospy.Subscriber("/vins_estimator/odometry", Odometry, self.odom_callback, queue_size=100)
        rospy.Subscriber("/gps", NavSatFix, self.gps_callback, queue_size=20)
        rospy.Subscriber("/baro_altitude", Float64, self.baro_callback, queue_size=50)

        self.init_particles(self.initial_s, self.initial_std_m, uniform=self.uniform_init_without_gps)
        rospy.loginfo("route_matcher loaded %d route points, length %.1f m from %s",
                      len(self.route.points), self.route.length, self.route_csv)

    def init_particles(self, center_s: float, std_m: float, uniform: bool = False):
        self.particles = []
        if uniform:
            for _ in range(self.num_particles):
                self.particles.append(random.uniform(0.0, self.route.length))
        else:
            center_s = clamp(center_s, 0.0, self.route.length)
            for _ in range(self.num_particles):
                self.particles.append(clamp(random.gauss(center_s, std_m), 0.0, self.route.length))
        self.weights = [1.0 / self.num_particles] * self.num_particles
        self.initialized = True

    def gps_callback(self, msg: NavSatFix):
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        x, y = lla_to_local_xy(msg.longitude, msg.latitude, self.route.lon0, self.route.lat0)
        sigma = self.sigma_gps_default
        cov = msg.position_covariance
        if len(cov) >= 5 and cov[0] > 0.0 and cov[4] > 0.0:
            sigma = max(math.sqrt(max(cov[0], cov[4])), 2.0)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self.latest_gps = (x, y, sigma, stamp)

        # If this is early and GPS is reliable, initialize around the nearest route point.
        if self.last_s_est is None or (not self.initialized):
            s0, d2 = self.route.nearest_by_xy(x, y)
            self.init_particles(s0, max(sigma * 2.0, 10.0), uniform=False)
            self.last_s_est = s0
            rospy.loginfo("route_matcher initialized by GPS: s=%.1f m, gps_dist=%.1f m", s0, math.sqrt(d2))
        # Publish state even without VINS odometry — GPS-only fallback.
        self.publish_state(msg.header.stamp if msg.header.stamp else rospy.Time.now())

    def baro_callback(self, msg: Float64):
        stamp = rospy.Time.now().to_sec()
        self.latest_baro_raw = (float(msg.data), stamp)
        eff = self.effective_baro_altitude()
        if eff is not None:
            self.baro_trace.append((stamp, eff))
            # Keep recent trace for slope estimate.
            cutoff = stamp - max(self.baro_history_window * 3.0, 30.0)
            self.baro_trace = [(t, z) for t, z in self.baro_trace if t >= cutoff]

    def effective_baro_altitude(self) -> Optional[float]:
        if self.latest_baro_raw is None:
            return None
        z_raw, _ = self.latest_baro_raw
        if self.baro_offset is None and self.auto_baro_offset and self.last_s_est is not None:
            rp = self.route.interpolate(self.last_s_est)
            self.baro_offset = rp.z - z_raw
            rospy.loginfo("baro offset calibrated: %.3f m", self.baro_offset)
        if self.baro_offset is None:
            return z_raw
        return z_raw + self.baro_offset

    def get_baro_delta(self, now_stamp: float) -> Optional[Tuple[float, float]]:
        """Return (delta_z, delta_time) over recent baro history."""
        if len(self.baro_trace) < 2:
            return None
        target = now_stamp - self.baro_history_window
        old = self.baro_trace[0]
        for item in self.baro_trace:
            if item[0] >= target:
                old = item
                break
        new = self.baro_trace[-1]
        dt = new[0] - old[0]
        if dt <= 1e-3:
            return None
        return new[1] - old[1], dt

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        current_pos = (pos.x, pos.y, pos.z)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        if self.last_odom_pos is None:
            self.last_odom_pos = current_pos
            self.last_odom_stamp = stamp
            self.publish_state(msg.header.stamp)
            return

        dx = current_pos[0] - self.last_odom_pos[0]
        dy = current_pos[1] - self.last_odom_pos[1]
        dz = current_pos[2] - self.last_odom_pos[2]
        delta_d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if delta_d > self.max_step_m:
            rospy.logwarn_throttle(2.0, "VIO delta %.2f m exceeds max_step_m %.2f; clipped", delta_d, self.max_step_m)
            delta_d = self.max_step_m
        self.last_delta_d = delta_d
        self.last_odom_pos = current_pos
        self.last_odom_stamp = stamp

        self.predict(delta_d)
        self.update_weights(stamp, delta_d)
        self.normalize_weights()
        if self.effective_sample_size() < self.resample_neff_ratio * self.num_particles:
            self.resample()
        self.last_s_est = self.estimate_s()
        self.last_state_stamp = stamp
        self.publish_state(msg.header.stamp)

    def predict(self, delta_d: float):
        for i, s in enumerate(self.particles):
            direction = 1.0
            if self.allow_backward:
                # Still biased forward; rarely allow small backward corrections.
                direction = 1.0 if random.random() > 0.03 else -1.0
            noisy_step = direction * delta_d + random.gauss(0.0, self.motion_noise)
            self.particles[i] = clamp(s + noisy_step, 0.0, self.route.length)

    def update_weights(self, stamp: float, delta_d: float):
        gps = self.latest_gps
        baro_z = self.effective_baro_altitude()
        baro_delta = self.get_baro_delta(stamp)
        # Decay stale GPS by ignoring it after 3 seconds.
        gps_valid = gps is not None and abs(stamp - gps[3]) < 3.0

        new_weights = []
        for s, old_w in zip(self.particles, self.weights):
            rp = self.route.interpolate(s)
            log_w = 0.0

            if gps_valid and gps is not None:
                gx, gy, sigma_gps, _ = gps
                d2 = (rp.x - gx) ** 2 + (rp.y - gy) ** 2
                # Soft gate: severe GPS mismatch is downweighted but not fully killed.
                sigma = max(sigma_gps, self.sigma_gps_default)
                gps_cost = min(d2 / (2.0 * sigma * sigma), (self.gps_gate_m ** 2) / (2.0 * sigma * sigma))
                log_w -= self.weight_gps * gps_cost

            if baro_z is not None:
                alt_err = rp.z - baro_z
                log_w -= self.weight_alt * (alt_err * alt_err) / (2.0 * self.sigma_alt * self.sigma_alt)

            if baro_delta is not None and self.last_s_est is not None:
                dz_baro, _ = baro_delta
                s_ref = max(0.0, s - max(delta_d, 1.0) * self.baro_history_window)
                rp_old = self.route.interpolate(s_ref)
                dz_route = rp.z - rp_old.z
                e = dz_route - dz_baro
                log_w -= self.weight_slope * (e * e) / (2.0 * self.sigma_slope * self.sigma_slope)

            # Motion continuity around previous estimate + VIO step.
            if self.last_s_est is not None:
                expected = clamp(self.last_s_est + delta_d, 0.0, self.route.length)
                e_motion = s - expected
                log_w -= self.weight_motion * (e_motion * e_motion) / (2.0 * self.sigma_motion * self.sigma_motion)

            # Avoid underflow.
            new_weights.append(math.exp(max(log_w, -80.0)) + 1e-300)
        self.weights = new_weights

    def normalize_weights(self):
        total = sum(self.weights)
        if not math.isfinite(total) or total <= 0.0:
            self.weights = [1.0 / self.num_particles] * self.num_particles
        else:
            self.weights = [w / total for w in self.weights]

    def effective_sample_size(self) -> float:
        return 1.0 / max(sum(w * w for w in self.weights), 1e-12)

    def resample(self):
        cumulative = []
        s = 0.0
        for w in self.weights:
            s += w
            cumulative.append(s)
        step = 1.0 / self.num_particles
        u = random.random() * step
        new_particles = []
        j = 0
        for _ in range(self.num_particles):
            while j < self.num_particles - 1 and u > cumulative[j]:
                j += 1
            new_particles.append(self.particles[j] + random.gauss(0.0, self.motion_noise * 0.4))
            u += step
        self.particles = [clamp(p, 0.0, self.route.length) for p in new_particles]
        self.weights = [1.0 / self.num_particles] * self.num_particles

    def estimate_s(self) -> float:
        return clamp(sum(s * w for s, w in zip(self.particles, self.weights)), 0.0, self.route.length)

    def estimate_std(self, mean_s: float) -> float:
        var = sum(w * (s - mean_s) ** 2 for s, w in zip(self.particles, self.weights))
        return math.sqrt(max(var, 0.0))

    def publish_state(self, stamp):
        if not self.particles:
            return
        s_est = self.estimate_s()
        self.last_s_est = s_est
        rp = self.route.interpolate(s_est)
        sigma_s = self.estimate_std(s_est)
        confidence = clamp(1.0 - sigma_s / 80.0, 0.0, 1.0)
        progress = 0.0 if self.route.length <= 0.0 else s_est / self.route.length
        baro_z = self.effective_baro_altitude()
        alt_err = None if baro_z is None else baro_z - rp.z
        gps_age = None
        if self.latest_gps is not None and self.last_odom_stamp is not None:
            gps_age = self.last_odom_stamp - self.latest_gps[3]

        state = {
            "s_m": round(s_est, 3),
            "route_length_m": round(self.route.length, 3),
            "progress": round(progress, 5),
            "segment_id": int(rp.segment_id),
            "distance_to_end_m": round(max(self.route.length - s_est, 0.0), 3),
            "matched_x": round(rp.x, 3),
            "matched_y": round(rp.y, 3),
            "matched_z": round(rp.z, 3),
            "matched_lon": round(rp.lon, 9),
            "matched_lat": round(rp.lat, 9),
            "yaw_rad": round(rp.yaw, 6),
            "s_std_m": round(sigma_s, 3),
            "confidence": round(confidence, 3),
            "last_vio_delta_m": round(self.last_delta_d, 3),
            "baro_altitude_m": None if baro_z is None else round(baro_z, 3),
            "altitude_error_m": None if alt_err is None else round(alt_err, 3),
            "gps_age_s": None if gps_age is None else round(gps_age, 3),
        }
        self.state_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))
        self.progress_pub.publish(Float64(data=progress))

        odom = Odometry()
        odom.header.stamp = stamp if stamp else rospy.Time.now()
        odom.header.frame_id = "map"
        odom.child_frame_id = "route_matcher"
        odom.pose.pose.position.x = rp.x
        odom.pose.pose.position.y = rp.y
        odom.pose.pose.position.z = rp.z
        # yaw to quaternion, roll/pitch zero
        half = 0.5 * rp.yaw
        odom.pose.pose.orientation.w = math.cos(half)
        odom.pose.pose.orientation.z = math.sin(half)
        # Simple covariance: uncertainty on route progress projected into position.
        cov = [0.0] * 36
        cov[0] = max(sigma_s * sigma_s, 0.01)
        cov[7] = max(sigma_s * sigma_s, 0.01)
        cov[14] = self.sigma_alt * self.sigma_alt
        cov[35] = 0.2
        odom.pose.covariance = cov
        self.odom_pub.publish(odom)


def main():
    rospy.init_node("route_matcher_node")
    ParticleFilterRouteMatcher()
    rospy.spin()


if __name__ == "__main__":
    main()
