from typing import List
import torch
import torch.nn as nn
from ..datasets.Sample import Sample
from .Detection import Detection

registered_detectors = dict()
class Detector(nn.Module):

    def __init__(self,num_channel:int, support_batch):
        super(Detector,self).__init__()
        self.num_channel = num_channel
        self.support_batch = support_batch
        self.device = torch.device("cpu")

    def forward(self, x:Sample, target=None, dataset=None) -> Detection:
        if dataset is None:
            from ..datasets.Coco import CocoDetection
            dataset = CocoDetection
        if not isinstance(x,Sample) and not isinstance(x,list):
            raise Exception("Argument is not a Sample")
        if isinstance(x,list):
            for v in x:
                if not isinstance(v,Sample):
                    raise Exception("Argument list contains non samples")
            if not self.support_batch:
                return [self.forward(v, dataset=dataset) for v in x]
            else:
                if self.num_channel ==1 :
                    return self._forward([v.getGray() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
                if self.num_channel ==3 :
                    return self._forward([v.getRGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
                if self.num_channel ==4 :
                    return self._forward([v.getARGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
        
        if self.num_channel ==1 :
            return self._forward(x.getGray(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        if self.num_channel ==3 :
            return self._forward(x.getRGB(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        if self.num_channel ==4 :
            return self._forward(x.getARGB(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        return x
    def adaptTo(self,dataset):
        print("Adapting to ",dataset.getName())
        return self
    def eval(self):
        pass
    def train(self):
        pass
    def save(self,file):
        torch.save(self.state_dict(),file)
    def load(self,file):
        state_dict= torch.load(file, map_location=self.device)
        self.load_state_dict(state_dict, strict = False)
    def to(self,device:torch.device):
        super(Detector,self).to(device)
        self.device = device
        return self
    def register(name:str,objClass):
        global registered_detectors
        registered_detectors[name]=objClass
    def getAllRegisteredDetectors():
        return dict(registered_detectors)
    def calculateLoss(self,sample:Sample):
        ret=self.forward(sample, sample)
        if isinstance(ret,list):
            ret= sum(ret)
        return ret

class GrayScaleDetector(Detector):
    def __init__(self):
        super(GrayScaleDetector,self).__init__(1,False)
    def _forward(self, gray:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        return Detection()

class TorchVisionDetector(Detector):
    def __init__(self, initiator,w, num_classes=None):
        super(TorchVisionDetector,self).__init__(3,True)
        self.initiator = initiator
        self.w=w
        self.model = initiator(weights=w)
        self.model.eval()
        self.dataset = "MS-COCO"
    def adaptTo(self,dataset):
        if self.dataset != dataset.getName():
            print("Torchvision model adapting to ",dataset.getName())
            newModel = TorchVisionDetector(self.initiator,self.w, num_classes=len(dataset.classesList())+1 )
            newModel.dataset = dataset.getName()
            return newModel
        else:
            return self
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        if isinstance(rgb,list):
            return self._forward(torch.cat([v.unsqueeze(0) for v in rgb],0), None,None, target)
        if len(rgb.shape)==3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.to(device=self.device)
        
        if(target is not None):
            if isinstance(target,list):
                target = [t.toTorchVisionTarget(self.device) for t in target]
            else:
                target = [target.toTorchVisionTarget(self.device)]
            loss_dict= self.model(rgb,target )
            return sum(loss for loss in loss_dict.values())
        torchvisionresult= self.model(rgb)
        return Detection.fromTorchVision(torchvisionresult, dataset)
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(TorchVisionDetector,self).to(device)
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
   
class TorchVisionInitiator():
    def __init__(self,initiator,w):
        self.initiator = initiator
        self.w=w
        pass
    def __call__(self):
        return TorchVisionDetector(self.initiator,self.w)
    
from functools import partial
import torchvision

Detector.register("fasterrcnn_mobilenet_v3_large_320_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1))
Detector.register("fasterrcnn_mobilenet_v3_large_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1))
Detector.register("fasterrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn, torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1))
#Detector.register("fasterrcnn_resnet50_fpn_v2",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn_v2, torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1))
Detector.register("fcos_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.fcos_resnet50_fpn, torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1))
#Detector.register("keypointrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.keypointrcnn_resnet50_fpn, torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1))
Detector.register("retinanet_resnet50_fpn_v2",TorchVisionInitiator(torchvision.models.detection.retinanet_resnet50_fpn_v2, torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1))
#Detector.register("maskrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.maskrcnn_resnet50_fpn, torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1))