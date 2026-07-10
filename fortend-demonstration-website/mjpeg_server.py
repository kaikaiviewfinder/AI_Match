#!/usr/bin/env python3
"""Simple MJPEG server for /vins_estimator/image_track"""
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

bridge = CvBridge()
latest_jpg = None
lock = threading.Lock()

def image_callback(msg):
    global latest_jpg
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        _, jpg = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            latest_jpg = jpg.tobytes()
    except Exception:
        pass

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while True:
                with lock:
                    jpg = latest_jpg
                if jpg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                rospy.sleep(0.05)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def main():
    rospy.init_node("mjpeg_server", anonymous=True)
    rospy.Subscriber("/vins_estimator/image_track", Image, image_callback)
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("MJPEG server on http://localhost:8080/stream")
    rospy.loginfo("MJPEG server on http://localhost:8080/stream")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rospy.spin()

if __name__ == "__main__":
    main()
