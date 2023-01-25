from interface.datasets import Sample
from interface.datasets.Coco import CocoDetection
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import math
from visualization_msgs.msg import MarkerArray, Marker
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from interface.detectors.Detector import Detector
from cv_bridge import CvBridge
import torch
import torchvision
import cv2
import threading
import time
class PredictionPub(Node):
    model : Detector
    def __init__(self) :
        super().__init__('detection_lidar_rviz')
        self.output = '/soul/roof/lidar/points/delayed'
        self.input =  '/soul/roof/lidar/points'
        self.sub = self.create_subscription(PointCloud2, self.input, self.callback_image, 10)
        self.pub = self.create_publisher(PointCloud2, self.output, 10)
        
    def callback_image(self, data:Image):
        print("_")
        def thread_function():
            time.sleep(2)
            self.pub.publish(data)
        data.header.stamp.sec += 2
        x = threading.Thread(target=thread_function)
        x.start()

def main (args=None):
    rclpy.init(args=args)
    Prediction_Pub = PredictionPub()
    rclpy.spin(Prediction_Pub)
    Prediction_Pub.destroy_node()
    rclpy.shutdown()
if __name__=='__main__' :
    main()


