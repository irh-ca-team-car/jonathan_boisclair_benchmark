from interface.datasets import Sample
from interface.datasets.detection.Cones import ConesDetection
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
    def __init__(self) :
        super().__init__('detection_rviz')
        self.output = '/apollo/prediction/perception_obstacles/visualizati_marker_array'
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
        #self.model : Detector = Detector.named("retinanet_resnet50_fpn_v2").to(device)
        self.dataset = ConesDetection()
        self.model : Detector = Detector.named("ssd_lite").adaptTo(self.dataset).to(device)

        try:
            self.model.load_state_dict(torch.load("cones.pth"))
        except:
            pass
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
        opencvImgrgb = opencvImg
        opencvImg=cv2.cvtColor(opencvImg, cv2.COLOR_BGR2HLS)

        #27%
        H = opencvImg[:,:,0]
        mask1 = cv2.inRange(H, 165, 180)
        mask2 = cv2.inRange(H, 0, 30)
        mask = mask1+mask1

        contours,h = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
        contours2 = []
        for cnt in contours:
            approx = cv2.approxPolyDP(
                cnt, 0.07 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3 and len(cnt) <6:
                print(cnt, len(cnt))
                contours2.append(cnt)
                #coordinates.append([cnt])
        opencvImg=cv2.drawContours(opencvImgrgb,contours2 ,-1, (0, 255, 0), 3)
                
        # height = 1.5
        # for f in det.boxes2d:
        #     obs = Marker()
        #     obs.header.frame_id = "car"
        #     if ConesDetection.getName(f.c) in ["cone"]:
        #         obs.type = Marker.CUBE
        #         obs.color.r = 0.0
        #         obs.color.g = 1.0
        #         obs.color.b = 0.0

        #         obs.scale.x = 1.0
        #         obs.scale.y = 1.0
        #         obs.scale.z = 0.3
        #         obs.color.a = 1.0
        #         height = 0.3
        #     else:
        #         continue
        #     obs.header.frame_id = "car"
        #     angleV = (f.h)*70.0 /s.size().h
        #     circ = 360 * height / angleV
        #     r = circ / (2*3.14)
        #     distance= -r
        #     obs.pose.position.y = float(distance)
        #     obs.pose.position.z = -0.8

        #     x = (f.x + f.w/2)
        #     half_width = s.size().w/2.0
        #     val = (x - half_width)/s.size().w
        #     angle = (150/2.0)*val
        #     obs.pose.position.x=float(math.tan(angle*0.01745329) * distance)
           
        #     obs.pose.orientation.w = 1.0
        #     obs.header.stamp = self.get_clock().now().to_msg()
            
        #     obs.action = obs.ADD
        #     #marker.pose.position.z = data.position.z
        #     obs.id = len(outputMsg.markers)
        #     outputMsg.markers.append(obs)

        self.running=False
        # self.pub.publish(outputMsg)
        # img = det.onImage(s)
        #opencvImg=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        imgMsg = CvBridge().cv2_to_imgmsg(opencvImg,"bgr8")
        self.pubviz.publish(imgMsg)
     
        imgMsg = CvBridge().cv2_to_compressed_imgmsg(opencvImg,"jpeg")
        self.pubvizcom.publish(imgMsg)

def main (args=None):
    rclpy.init(args=args)
    Prediction_Pub = PredictionPub()
    rclpy.spin(Prediction_Pub)
    Prediction_Pub.destroy_node()
    rclpy.shutdown()
if __name__=='__main__' :
    main()


