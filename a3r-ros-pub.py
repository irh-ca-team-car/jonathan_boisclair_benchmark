# -*- coding: utf-8 -*-
"""cv_bridge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zPypXgSr9cxqUEPiTOZunDMR6iE1BTHD
"""

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
from rclpy.node import Node


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('detection_rviz')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image,"/output_image", 1)
        self.timer = self.create_timer(0.1,self.tmr)
        self.vid = cv2.VideoCapture(0)

    def tmr(self):
        ret, frame = self.vid.read()
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    

if __name__ == '__main__':
    rclpy.init()
    ip = ImagePublisher()

    rclpy.spin(ip)