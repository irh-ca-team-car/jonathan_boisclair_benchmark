# -*- coding: utf-8 -*-
"""cv_bridge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zPypXgSr9cxqUEPiTOZunDMR6iE1BTHD
"""

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import cv2
from rclpy.node import Node
from threading import Thread
import os
import time
class VideoStream:
    """Class for CV2 video capture. The start() method will create a new 
thread to read the video stream"""
    def __init__(self, src=0):
        
        
        # self._boxes = None
        self.stopped = False
        self.devices = set([f for f in os.listdir("/dev") if "video" in f])
        self.src = "/dev/"+list(self.devices)[-1]
        self.stream = cv2.VideoCapture(self.src)
        print("Opening",self.src)
        (self.grabbed, self.frame) = self.stream.read()


    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            devices = set([f for f in os.listdir("/dev") if "video" in f])
            newDevices = devices-self.devices
            if(len(newDevices)) > 0:
                time.sleep(1)
                devices = set([f for f in os.listdir("/dev") if "video" in f])
                newDevices = devices-self.devices
                if(len(newDevices)) > 0:
                    lst = list(newDevices)
                    lst.sort()
                    self.src = "/dev/"+ lst[0]
                    print("New device found, opening", self.src, newDevices)
                    time.sleep(1)
                    self.stream = cv2.VideoCapture(self.src)
                    (self.grabbed, self.frame) = self.stream.read()
            self.devices = devices
            try:
                #if not self.stream.isOpened():
                    #print("Device unplugged, trying to reconnect")
                    #self.stream = cv2.VideoCapture(self.src)
                (self.grabbed, self.frame) = self.stream.read()
            except:
                try:
                    self.stream = cv2.VideoCapture(self.src)
                except:
                    print("Waiting for a new device")
                    time.sleep(1)
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
        self.image_pub_com = self.create_publisher(CompressedImage,"/output_image/compressed", 1)
        self.timer = self.create_timer(1/30.0,self.tmr)
        #self.vid = cv2.VideoCapture("/dev/video0")

    def tmr(self):
        frame = self.exchange.frame
        if frame is not None and frame.size >0:
            try:
                width,height = self.exchange.get_video_dimensions()
                left = frame[:,0:int(width/2)]
                right = frame[:,int(width/2):width]
                frame = cv2.scaleAdd(left,0.5,right)
                frame = cv2.resize(left, (480, 352))
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
                self.image_pub_com.publish(self.bridge.cv2_to_compressed_imgmsg(frame, "jpeg"))
            except:
                pass

if __name__ == '__main__':
    rclpy.init()
    
    exchange = VideoStream().start()

    ip = ImagePublisher(exchange)

    rclpy.spin(ip)

    exchange.stop_process()