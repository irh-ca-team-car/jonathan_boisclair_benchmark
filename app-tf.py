from interface.datasets import Sample
from interface.datasets.Coco import CocoDetection
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import math
from visualization_msgs.msg import MarkerArray, Marker
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from interface.detectors.Detector import Detector
from cv_bridge import CvBridge
import torch
import torchvision
import cv2
import json
import yaml
import os
import shutil
class PredictionPub(Node):
    model : Detector
    def __init__(self) :
        super().__init__('apollo_transform_publisher')
        self.timer = self.create_timer(0.1, self.tmr)
        self.pub = self.create_publisher(TFMessage,"/tf",1)
        self.pubs = self.create_publisher(TFMessage,"/tf_static",1)
        self.tfmsg = TFMessage()
        f = open("tf/transform.json")
        self.transform = json.load(f)
        f.close()

        for obj in self.transform["extrinsicfile"]:
            file = obj["filepath"]

            filePath = "tf/"+file
            if not os.path.exists(filePath):
                for root, dirs, files in os.walk("/home/boiscljo/ros2apollo/src", topdown=False):
                    for name in files:
                        if name == file:
                            shutil.copy(os.path.join(root, name), filePath)
                pass

            f = open(filePath)
            yamlObj = yaml.safe_load(f)
            transform = TransformStamped()
            transform.header.frame_id = yamlObj["header"]["frame_id"]
            transform.child_frame_id = yamlObj["child_frame_id"]
            transform.transform.rotation.x =float(yamlObj["transform"]["rotation"]["x"])
            transform.transform.rotation.y =float(yamlObj["transform"]["rotation"]["y"])
            transform.transform.rotation.z =float(yamlObj["transform"]["rotation"]["z"])
            transform.transform.rotation.w =float(yamlObj["transform"]["rotation"]["w"])
            transform.transform.translation.x =float(yamlObj["transform"]["translation"]["x"])
            transform.transform.translation.y =float(yamlObj["transform"]["translation"]["y"])
            transform.transform.translation.z =float(yamlObj["transform"]["translation"]["z"])

            self.tfmsg.transforms.append(transform)
            print(yamlObj)

            f.close()
    def tmr(self):
        for stamped in self.tfmsg.transforms:
            stamped : TransformStamped
            stamped.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.tfmsg)
        self.pubs.publish(self.tfmsg)
        pass

def main (args=None):
    rclpy.init(args=args)
    Prediction_Pub = PredictionPub()
    rclpy.spin(Prediction_Pub)
    Prediction_Pub.destroy_node()
    rclpy.shutdown()
if __name__=='__main__' :
    main()


