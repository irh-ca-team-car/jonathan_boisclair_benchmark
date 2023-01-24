
from .Detector import Detector
from ..datasets import Sample, Box2d,Detection
import torch
import torchvision.transforms
class YoloV5Detector(Detector):
    model:torch.nn.Module
    def __init__(self,  **kwarg):
        super(YoloV5Detector,self).__init__(3,True)
        self.kwarg = kwarg
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True, autoshape=True)
        self.model.eval()
        self.dataset = "MS-COCO"
    def adaptTo(self,dataset):
        if self.dataset != dataset:
            print("Torchvision model adapting to ",dataset.getName())
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True, autoshape=True, classes=len(dataset.classesList()))
            #newModel = TorchVisionDetector(self.initiator,num_classes=len(dataset.classesList(), self.kwarg) )
            #try:
            #    newModel.load_state_dict(self.state_dict(),strict=False)
            #except:
            #    pass
            #print("New dataset has ",len(dataset.classesList()),"classes")
            #newModel.dataset = dataset
            #return newModel
            return self
        else:
            return self
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        if isinstance(rgb,list) and rgb[0].__class__.__name__ != "Image":
            return self._forward([torchvision.transforms.ToPILImage()(v) for v in rgb], None,None, target)
        if not isinstance(rgb,list) and rgb.__class__.__name__ != "Image":
            rgb = torchvision.transforms.ToPILImage()(rgb)
        pandasresult = self.model(rgb).pandas().xyxy

        result = []
        for pandas in pandasresult:
            detection = Detection()
            for row in pandas.itertuples():
                box = Box2d()
                box.x = row.xmin
                box.y = row.ymin
                box.w = row.xmax-row.xmin
                box.h = row.xmax-row.ymin
                box.c = row._6
                box.cf = row.confidence
                box.cn = row.name
                detection.boxes2d.append(box)
            result.append(detection)
        if not isinstance(rgb,list):
            return result[0]
        return result
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(YoloV5Detector,self).to(device)
        self.model = self.model.to(device)
        return self
    def calculateLoss(self,sample:Sample):
        #if isinstance(sample,list):
        #    ret=sum(self.forward(sample, sample))
        #else:
        #    ret= self.forward(sample, sample)
        #return ret

        losses= self.forward(sample, sample)
        return losses
   