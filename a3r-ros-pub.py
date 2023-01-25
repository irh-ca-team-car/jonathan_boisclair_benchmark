# -*- coding: utf-8 -*-
"""cv_bridge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zPypXgSr9cxqUEPiTOZunDMR6iE1BTHD
"""

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from rclpy.node import Node
from threading import Thread

class VideoStream:
    """Class for CV2 video capture. The start() method will create a new 
thread to read the video stream"""
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # self._boxes = None
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            try:
                if not self.stream.isOpened():
                    self.stream = cv2.VideoCapture(self.src)
                (self.grabbed, self.frame) = self.stream.read()
            except:
                self.stream = cv2.VideoCapture(self.src)
    def get_fps(self):
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        return (fps)
    def get_video_dimensions(self):
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def stop_process(self):
        self.stopped = True


class ImagePublisher(Node):
    def __init__(self,exchange:VideoStream):
        super().__init__('detection_rviz')
        self.exchange = exchange
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image,"/output_image", 1)
        self.timer = self.create_timer(1/exchange.get_fps(),self.tmr)
        #self.vid = cv2.VideoCapture("/dev/video0")

    def tmr(self):
        frame = self.exchange.frame
        if frame.size >0:
            width,height = self.exchange.get_video_dimensions()
            frame = frame[:,0:int(width/2)]
            frame = cv2.resize(frame, (480, 352))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))


if __name__ == '__main__':
    rclpy.init()
    
    exchange = VideoStream(0).start()

    ip = ImagePublisher(exchange)

    rclpy.spin(ip)

    exchange.stop_process()