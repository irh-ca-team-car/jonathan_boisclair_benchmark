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

    def forward(self, x:Sample):
        if not isinstance(x,Sample) and not isinstance(x,list):
            raise Exception("Argument is not a Sample")
        if isinstance(x,list):
            for v in x:
                if not isinstance(v,Sample):
                    raise Exception("Argument list contains non samples")
            if not self.support_batch:
                return [self.forward(v) for v in x]
            else:
                if self.num_channel ==1 :
                    return self._forward([v.getGray() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x])
                if self.num_channel ==3 :
                    return self._forward([v.getRGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x])
                if self.num_channel ==4 :
                    return self._forward([v.getARGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x])
        
        if self.num_channel ==1 :
            return self._forward(x.getGray(), x.getLidar(), x.getThermal())
        if self.num_channel ==3 :
            return self._forward(x.getRGB(), x.getLidar(), x.getThermal())
        if self.num_channel ==4 :
            return self._forward(x.getARGB(), x.getLidar(), x.getThermal())
        return x
    def register(name:str,objClass):
        global registered_detectors
        registered_detectors[name]=objClass
    def getAllRegisteredDetectors():
        return dict(registered_detectors)
        

class GrayScaleDetector(Detector):
    def __init__(self):
        super(GrayScaleDetector,self).__init__(1,False)
    def _forward(self, gray:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor):
        return Detection()

class TorchVisionDetector(Detector):
    def __init__(self, initiator):
        super(TorchVisionDetector,self).__init__(3,True)
        self.model = initiator()
        self.model.eval()
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor):
        if isinstance(rgb,list):
            return self._forward(torch.cat([v.unsqueeze(0) for v in rgb],0), None,None)
        if len(rgb.shape)==3:
            rgb = rgb.unsqueeze(0)
        torchvisionresult= self.model(rgb)
        return Detection.fromTorchVision(torchvisionresult)
    
class TorchVisionInitiator():
    def __init__(self,initiator):
        self.initiator = initiator
        pass
    def __call__(self):
        return TorchVisionDetector(self.initiator)

from functools import partial
import torchvision

Detector.register("GrayScaleDetector",GrayScaleDetector)
Detector.register("FRNN",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn))