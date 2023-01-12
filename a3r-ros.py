from interface.datasets import Sample
from interface.datasets.Coco import CocoDetection
import rclpy
from rclpy.node import Node
import math
from apollo_msgs.msg import ApolloperceptionPerceptionObstacles,ApolloperceptionPerceptionObstacle
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from interface.detectors.Detector import Detector
from cv_bridge import CvBridge
import torch
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
        self.output = '/apollo/perception/obstacles'
        self.input =  '/zed_wrapper_fl/left/image_rect_color'
        self.sub = self.create_subscription(Image, self.input, self.callback_image, 10)
        self.input2 =  '/zed_wrapper_front/left/image_rect_color'
        self.sub = self.create_subscription(Image, self.input2, self.callback_image, 10)
        self.pub = self.create_publisher(ApolloperceptionPerceptionObstacles, self.output, 10)
        self.pubviz = self.create_publisher(Image, "/apollo/perception/obstacles/viz", 10)
        self.pubvizcom = self.create_publisher(CompressedImage, "/apollo/perception/obstacles/viz/com", 10)
        self.stamp = Header().stamp
        #self.model : Detector = Detector.named("retinanet_resnet50_fpn_v2").to(device)
        self.model : Detector = Detector.named("ssd_lite").to(device)
        print(self.model.device)
        self.running=False
        
    def callback_image(self, data:Image):
        if self.running:
            return
        self.running=True
        opencvImg = CvBridge().imgmsg_to_cv2(data, "bgr8")
        tensor = torch.from_numpy(opencvImg)
        b = tensor[:,:,0].unsqueeze(0)
        g = tensor[:,:,1].unsqueeze(0)
        r = tensor[:,:,2].unsqueeze(0)

        s = Sample()
        s.setImage((torch.cat([r,g,b],0).float()/255).to(device))
        det = self.model.forward(s).filter(0.25).NMS_Pytorch()

        outputMsg = ApolloperceptionPerceptionObstacles()
        for f in det.boxes2d:
            obs = ApolloperceptionPerceptionObstacle()
            if CocoDetection.getName(f.c) in ["person"]:
                obs.type=3 
            elif CocoDetection.getName(f.c) in ["bicycle","motorcycle"]:
                obs.type=4
            elif CocoDetection.getName(f.c) in ["car","truck","train","bus"]:
                obs.type=5 
            elif CocoDetection.getName(f.c) in ["stop sign","fire hydrant"]:
                obs.type = 0
            else:
                continue
            obs.position.x =0.0
            obs.position.y =0.0
            
            angleV = (f.h)*70.0 /s.size().h
            circ = 360 * 1.5 / angleV
            r = circ / (2*3.14)
            distance= -r
            obs.position.y = float(distance)

            x = (f.x + f.w/2)
            half_width = s.size().w/2.0
            val = (x - half_width)/s.size().w
            angle = (150/2.0)*val
            obs.position.x=float(math.tan(angle*0.01745329) * distance);
                
            outputMsg.perceptionobstacle.append(obs)

        self.running=False
        self.pub.publish(outputMsg)
        img = det.onImage(s)
        imgMsg = CvBridge().cv2_to_imgmsg(tensorToCV2(img),"bgr8")
        self.pubviz.publish(imgMsg)

def main (args=None):
    rclpy.init(args=args)
    Prediction_Pub = PredictionPub()
    rclpy.spin(Prediction_Pub)
    Prediction_Pub.destroy_node()
    rclpy.shutdown()
if __name__=='__main__' :
    main()


