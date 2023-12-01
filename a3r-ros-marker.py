from interface.datasets import Sample
from interface.datasets.detection.Coco import CocoDetection
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
import zstd
import cv2
from apollo_msgs.msg import ApolloperceptionPerceptionObstacles,ApolloperceptionPerceptionObstacle

def tensorToCV2(t:torch.Tensor):
    if len(t.shape) ==4:
        t=t[0]
    t = t.cpu().permute(1, 2, 0)
    np_ = t.detach().numpy()
    np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
    return np_
def show(t: torch.Tensor,wait: bool = False):
    if len(t.shape) ==3:
        t=t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, scale_factor=(1.0,1.0))
    if len(t.shape) ==4:
        t=t[0]
    t = t.cpu().permute(1, 2, 0)
    np_ = t.detach().numpy()
    np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", np_)
    # for i in range(30):

    if wait:
        while True:
            cv2.imshow("Image", np_)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break
    else:
        cv2.waitKey(1)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class PredictionPub(Node):
    model : Detector
    #General method to gather params --ros-args values
    def param(self, name, default_value=None):
        node: rclpy.node.Node = self
        print(name, default_value)
        if not self.has_parameter(name):
            self.declare_parameter(name)
        if(default_value == None):
            if(self.has_parameter(name)):
                return self.get_parameter(name).value
            return self.get_parameter_or(name, rclpy.Parameter(name, rclpy.Parameter.Type.STRING, "")).value
        if default_value.__class__ == bool:
            param_type = rclpy.Parameter.Type.BOOL
        elif default_value.__class__ == float:
            param_type = rclpy.Parameter.Type.DOUBLE
        elif default_value.__class__ == int:
            param_type = rclpy.Parameter.Type.INTEGER
        else:  # default_value.__class__ == str:
            param_type = rclpy.Parameter.Type.STRING
            default_value = str(default_value)

        value = self.get_parameter_or(name, rclpy.Parameter(
            name, param_type, default_value)).value
        if value is None:
            return default_value
        return value
    def __init__(self) :
        super().__init__('detection_rviz')
        self.output = '/apollo/prediction/perception_obstacles/visualizati_marker_array'
        self.apollo_topic = self.param("apollo_topic", '/apollo/perception/obstacles/local')
        self.input =  '/zed_wrapper_fl/left/image_rect_color'
        self.sub = self.create_subscription(Image, self.input, self.callback_image, 10)
        self.input2 =  '/zed_wrapper_front/left/image_rect_color'
        self.sub2 = self.create_subscription(Image, self.input2, self.callback_image, 10)
        self.input2 =  '/output_image/compressed'
        self.sub3 = self.create_subscription(CompressedImage, self.input2, self.callback_image, 10)
        self.pub = self.create_publisher(MarkerArray, self.output, 10)
        self.pubviz = self.create_publisher(Image, "/apollo/perception/obstacles/viz", 10)
        self.pubvizcom = self.create_publisher(CompressedImage, "/apollo/perception/obstacles/viz/compressed", 10)
        self.stamp = Header().stamp
        self.pub_apollo = self.create_publisher(ApolloperceptionPerceptionObstacles, self.apollo_topic, 10)

        #self.model : Detector = Detector.named("retinanet_resnet50_fpn_v2").to(device)
        self.model : Detector = Detector.named("ssd_lite").to(device)
        print(self.model.device)
        self.running=False
        
    def callback_image(self, data:Image):
        
        with torch.no_grad():
            pass
        if self.running:
            return
        self.running=True
        if isinstance(data,CompressedImage):
            opencvImg=CvBridge().compressed_imgmsg_to_cv2(data,"bgr8")
        else:
            opencvImg = CvBridge().imgmsg_to_cv2(data, "bgr8")
        tensor = torch.from_numpy(opencvImg)
        b = tensor[:,:,0].unsqueeze(0)
        g = tensor[:,:,1].unsqueeze(0)
        r = tensor[:,:,2].unsqueeze(0)

        s = Sample()
        s.setImage((torch.cat([r,g,b],0).float()/255).to(device))
        det = self.model.forward(s).filter(0.25).NMS_Pytorch()

        
        apolloMsg = ApolloperceptionPerceptionObstacles()
        outputMsg = MarkerArray()

        obs = Marker()
        obs.id = 0
        obs.header.frame_id = "car"
        obs.action = Marker.DELETEALL
        outputMsg.markers.append(obs)

        height = 1.5
        for f in det.boxes2d:
            obs = Marker()
            obs.header.frame_id = "car"
            if CocoDetection.getName(f.c) in ["person"]:
                obs.type = Marker.MESH_RESOURCE
                obs.mesh_resource = "package://kia_soul/Man_with_suit.stl"
                obs.color.r = 0.0
                obs.color.g = 1.0
                obs.color.b = 0.0
                height = 1.5
                if f.w > f.h:
                    height = 0.75
                #obs.type=3 
            elif CocoDetection.getName(f.c) in ["bicycle","motorcycle"]:
                obs.type = obs.CUBE
                obs.color.r = 0.0
                obs.color.g = 1.0
                obs.color.b = 1.0
                height = 2.5
                #obs.type=4
            elif CocoDetection.getName(f.c) in ["car","truck","bus"]:
                obs.type = Marker.MESH_RESOURCE
                obs.mesh_resource = "package://kia_soul/soul.dae"
                obs.color.r = 0.0
                obs.color.g = 1.0
                obs.color.b = 0.0
                height = 3
                #obs.type=5 
            elif CocoDetection.getName(f.c) in ["stop sign","fire hydrant"]:
                obs.type = obs.CUBE
                obs.color.r = 1.0
                obs.color.g = 0.0
                obs.color.b = 0.0
                height = 0.5
                #obs.type = 0
            else:
                continue
            obs.header.frame_id = "car"
            angleV = (f.h)*70.0 /s.size().h
            circ = 360 * height / angleV
            r = circ / (2*3.14)
            distance= -r
            obs.pose.position.y = float(distance)

            x = (f.x + f.w/2)
            half_width = s.size().w/2.0
            val = (x - half_width)/s.size().w
            angle = (150/2.0)*val
            obs.pose.position.x=float(math.tan(angle*0.01745329) * distance);
                
            obs.scale.x = 1.0
            obs.scale.y = 1.0
            obs.scale.z = 1.0
            obs.color.a = 1.0
           
            obs.pose.orientation.w = 1.0
            obs.header.stamp = self.get_clock().now().to_msg()
            
            obs.action = obs.ADD
            #marker.pose.position.z = data.position.z
            obs.id = len(outputMsg.markers)
            outputMsg.markers.append(obs)
            #apollo format
            obs_apollo = ApolloperceptionPerceptionObstacle()
            if CocoDetection.getName(f.c) in ["person"]:
                obs_apollo.type=3 
            elif CocoDetection.getName(f.c) in ["bicycle","motorcycle"]:
                obs_apollo.type=4
            elif CocoDetection.getName(f.c) in ["car","truck","train","bus"]:
                obs_apollo.type=5 
            elif CocoDetection.getName(f.c) in ["stop sign","fire hydrant"]:
                obs_apollo.type = 0

            obs_apollo.position.x = obs.pose.position.x
            obs_apollo.position.y = obs.pose.position.y

            apolloMsg.perceptionobstacle.append(obs)

        self.running=False
        self.pub_apollo.publish(apolloMsg)
        self.pub.publish(outputMsg)
        img = det.onImage(s)
        imgMsg = CvBridge().cv2_to_imgmsg(tensorToCV2(img),"bgr8")
        self.pubviz.publish(imgMsg)
     
        imgMsg = CvBridge().cv2_to_compressed_imgmsg(tensorToCV2(img),"jpeg")
        self.pubvizcom.publish(imgMsg)

def main (args=None):
    rclpy.init(args=args)
    Prediction_Pub = PredictionPub()
    rclpy.spin(Prediction_Pub)
    Prediction_Pub.destroy_node()
    rclpy.shutdown()
if __name__=='__main__' :
    main()


