from typing import Callable, Dict, List, Union
import torch
import torch.nn as nn
from ..datasets.Sample import Sample
from .Detection import Detection


class Detector(nn.Module):
    registered_detectors: Dict[str,Callable[[],"Detector"]] = dict()
    def __init__(self,num_channel:int, support_batch):
        super(Detector,self).__init__()
        self.num_channel = num_channel
        self.support_batch = support_batch
        self.device = torch.device("cpu")

    def forward(self, x:Sample, target=None, dataset=None) -> Detection:
        if dataset is None:
            from ..datasets.detection import CocoDetection
            dataset = CocoDetection()
        if not isinstance(x,Sample) and not isinstance(x,list):
            raise Exception("Argument is not a Sample")
        if isinstance(x,list):
            for v in x:
                if not isinstance(v,Sample):
                    raise Exception("Argument list contains non samples")
            if not self.support_batch:
                if target is not None:
                    return sum([self.forward(v[0],target=v[1], dataset=dataset) for v in zip(x,target)])
                
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
    def adaptTo(self,dataset) -> "Detector":
        print("Adapting to ",dataset.getName())
        return self
    def eval(self) -> "Detector":
        return self
    def train(self) -> "Detector":
        return self
    def save(self,file) ->None:
        torch.save(self.state_dict(),file)
    def load(self,file)->None:
        state_dict= torch.load(file, map_location=self.device)
        try:
            self.load_state_dict(state_dict, strict = False)
        except:
            pass
    def to(self,device:torch.device) -> "Detector":
        super(Detector,self).to(device)
        self.device = device
        return self
    def register(name:str,objClass) -> None:
        Detector.registered_detectors[name]=objClass
    def named(name:str) -> "Detector":
        c= Detector.registered_detectors[name]()
        return c
    def getAllRegisteredDetectors() -> Dict[str,Callable[[],"Detector"]]:
        return dict(Detector.registered_detectors)
    def calculateLoss(self,sample:Union[Sample,List[Sample]])->torch.Tensor:
        ret=self.forward(sample, sample)
        if isinstance(ret,list):
            ret= sum(ret)
        return ret
    @staticmethod
    def optimizer(model):
        return torch.optim.Adamax(model.parameters())

class TorchVisionDetector(Detector):
    model:torch.nn.Module
    def __init__(self, initiator, num_classes=None, **kwarg):
        super(TorchVisionDetector,self).__init__(3,True)
        self.initiator = initiator
        self.kwarg = kwarg
        if num_classes is not None:
            self.model:torch.nn.Module = initiator(num_classes=num_classes, **self.kwarg)
        else:
            self.model = initiator(**self.kwarg)
        self.model.eval()
        self.dataset = "MS-COCO"
    def adaptTo(self,dataset):
        if self.dataset != dataset:
            if "weights" in self.kwarg:
                del self.kwarg["weights"]
            if "pretrained" in self.kwarg:
                del self.kwarg["pretrained"]
            newModel = TorchVisionDetector(self.initiator,num_classes=len(dataset.classesList()), **self.kwarg )
            try:
                newModel.load_state_dict(self.state_dict(),strict=False)
            except:
                pass
            newModel.dataset = dataset
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
        losses= self.forward(sample, sample)
        return losses
   
class TorchVisionInitiator():
    def __init__(self,initiator, **kwarg):
        self.initiator = initiator
        self.kwarg = kwarg
        pass
    def __call__(self):
        return TorchVisionDetector(self.initiator,**self.kwarg)
    
from functools import partial
import torchvision

try:
    Detector.register("fasterrcnn_mobilenet_v3_large_320_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1))
    Detector.register("fasterrcnn_mobilenet_v3_large_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1))
    Detector.register("fasterrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn, weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1))
    Detector.register("fcos_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.fcos_resnet50_fpn, weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1))
    Detector.register("retinanet_resnet50_fpn_v2",TorchVisionInitiator(torchvision.models.detection.retinanet_resnet50_fpn_v2, weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1))
    Detector.register("ssd",TorchVisionInitiator(torchvision.models.detection.ssd300_vgg16, weights=torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1))
    Detector.register("ssd_lite",TorchVisionInitiator(torchvision.models.detection.ssdlite320_mobilenet_v3_large, weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.COCO_V1))
except BaseException as e:
    def format_exception(e):
        import sys
        import traceback

        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

        exception_str = "Traceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        # Removing the last \n
        exception_str = exception_str[:-1]

        return exception_str
    print("Seems like weights are not implemented, are you using a old pytorch version?",format_exception(e))
    Detector.register("fasterrcnn_mobilenet_v3_large_320_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, pretrained=True))
    Detector.register("fasterrcnn_mobilenet_v3_large_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, pretrained=True))
    Detector.register("fasterrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn, pretrained=True))
    Detector.register("ssd",TorchVisionInitiator(torchvision.models.detection.ssd300_vgg16, pretrained=True))
    Detector.register("ssd_lite",TorchVisionInitiator(torchvision.models.detection.ssdlite320_mobilenet_v3_large, pretrained=True))
   
#Incompatible for some reasons
#Detector.register("fasterrcnn_resnet50_fpn_v2",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn_v2, torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1))
#Detector.register("keypointrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.keypointrcnn_resnet50_fpn, torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1))
#Detector.register("maskrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.maskrcnn_resnet50_fpn, torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1))
