from typing import Any, Iterator, Mapping
from ..datasets.Sample import Size
from ..detectors.Detector import *
from ..detectors.Detection import *
from ..datasets.detection import DetectionDataset
import torch

import argparse
import time
from pathlib import Path
import argparse
import sys
import time
import warnings
import pathlib
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ..transforms import ScaleTransform, RequiresGrad, Cat
from ..transforms import apply as tf

yolov8_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "yolov8")
sys.path.append(pathlib.Path(__file__).parent.absolute().as_posix())
sys.path.append("./yolov8")
sys.path.append("./interface/impl/yolov8")
os.environ["PYTHONPATH"]=  os.environ.get("PYTHONPATH")+":"+yolov8_dir

from .yolov8.ultralytics import YOLO


class hyper():
    def __init__(self) -> None:
        self.min_memory=0
        self.box=0.05
        self.cls=0.5
        self.dfl=0
class YoloV8Detector(Detector):
    model: YOLO
    dataset: DetectionDataset
    isTrain: bool

    def __init__(self,mdl="yolov8n") -> None:
        super(YoloV8Detector, self).__init__(3, False)
        self.device = "cpu"
        from yolov8.ultralytics.yolo.v8.detect.train import DetectionTrainer
        self.model = YOLO(mdl+".pt").train()
        self.det = self.model.model
        self.model_name = mdl
        self.trainer = self.model.trainer
        self.dataset = DetectionDataset.named("coco-empty")
        self.model.task = "detect"
        self.adaptTo(self.dataset)

    def train(self,train=True) -> "YoloV8Detector":
        super().train()
        self.det.train(train)
        return self
    def eval(self) -> "YoloV8Detector":
        super().eval()
        self.det.eval()
        return self
    def to(self,device:torch.device):
        super(YoloV8Detector,self).to(device)
        self.model.to(device)
        self.det.to(device)
        self.device=device
        self.model.overrides["device"]=device
        return self
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        
        if isinstance(rgb,list):
            return self._forward(torch.cat([v.unsqueeze(0) for v in rgb],0), None,None, target)
        if len(rgb.shape)==3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.to(device=self.device).clone()
        if isinstance(rgb,list) and rgb[0].__class__.__name__ != "Image":
            return self._forward([torchvision.transforms.ToPILImage()(v) for v in rgb], None,None, target)
        if not isinstance(rgb,list) and rgb.__class__.__name__ != "Image":
            rgb = torchvision.transforms.ToPILImage()(rgb[0])

        with torch.autocast(device_type="cpu" if "cpu" in str(self.device) else "cuda",enabled=True):
            pred = self.model.predict(rgb, augment=False, verbose=False, conf=0.00000005)

        result = []
        for pred_ in pred:
            detection = Detection()
            dboxes = pred_.boxes
            sizes = pred_.boxes.cpu().xywh
            conf =  pred_.boxes.cpu().conf
            cls =  pred_.boxes.cpu().cls
            for b in range(dboxes.shape[0]):
                box = Box2d()
                box.c = int(dboxes[b].cls) 
                box.cf = float(dboxes[b].conf)
                box.cn = self.dataset.getName(box.c)
              
                box.w = float(sizes[b][2])
                box.h = float(sizes[b][3])

                box.x = float(sizes[b][0]) - box.w/2
                box.y = float(sizes[b][1]) - box.h/2
                detection.boxes2d.append(box)

            result.append(detection)
        if not isinstance(rgb,list):
            return result[0]
        return result
    @staticmethod
    def optimizer(model):
        from yolov8.ultralytics.yolo.engine.trainer import BaseTrainer
        return BaseTrainer.build_optimizer(model)
    def freeze_backbone(self):
        for name,p in (self.named_parameters()):
            if ".cv2." not in name and ".cv3." not in name:
                p.requires_grad_(False)
        
    def unfreeze_backbone(self):
        for name,p in self.named_parameters():
            p.requires_grad(True)
  
    def adaptTo(self, dataset):

        nc = len(dataset.classesList())
        from .yolov8.ultralytics.nn.tasks import DetectionModel
        state_dict = self.det.state_dict()
        self.model.model = DetectionModel(self.model_name+".yaml",ch=3, nc=nc,verbose=False)
        self.det = self.model.model
        self.det.names = dataset.classesList()

        try:
            self.det.load_state_dict(state_dict,strict=False)
        except Exception as e:
            pass
        self.dataset = dataset
        return self
   
    def calculateLoss(self,sample:Sample):
        self.train()

        from .yolov8.ultralytics.yolo.v8.detect.train import Loss
    
        self.det.args=hyper()
        self.loss = Loss(self.det)

        #for param in self.det.parameters():
        #    param.requires_grad = True # or True
        rgb = sample
        if isinstance(sample, Sample):
            sample = [sample]
            rgb = [rgb]
        transforms = [Cat(), RequiresGrad(True) ,self.device]
        sample = ScaleTransform(Size(640,640))(sample)
        rgb = tf(sample,transforms)
        with torch.autocast(device_type="cpu" if "cpu" in str(self.device) else "cuda",enabled=True):
            tensorOut = self.det(rgb)
            targets= []
            for img in range(rgb.shape[0]):
                detectionImages : Detection = sample[img].detection
                for box in detectionImages.boxes2d:
                    factor=640
                    bx= [
                        img, box.c, (box.x+box.w/2)/factor,(box.y+box.h/2)/factor,box.w/factor,box.h/factor
                    ]
                    if box.x < 0: continue
                    if box.y < 0: continue
                    if box.w < 10: continue
                    if box.h < 10: continue
                    if box.x+box.w >=640: continue
                    if box.y+box.h >=640: continue
                    targets.append(bx)

            if len(targets)>0:
                targets = torch.tensor(targets).float()
            else:
                targets = torch.Tensor(0,6)#image,class,x,y,w,h

            losses, loss_items = self.loss.__call__(tensorOut, targets.to(self.device))

        self.eval()
        return losses
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        self.det = state_dict
        self.model.model = self.det
    def state_dict(self):
        return self.det
    
    # def modules(self) -> Iterator[torch.nn.Module]:
    #     return self.det.modules()

  
class YoloV8DetectorInitiator():
    def __init__(self,coef):
        self.coef=coef
        pass
    def __call__(self):
        return YoloV8Detector(self.coef)

Detector.register("yolov8n",YoloV8DetectorInitiator('yolov8n'))
Detector.register("yolov8l",YoloV8DetectorInitiator('yolov8l'))
Detector.register("yolov8m",YoloV8DetectorInitiator('yolov8m'))
Detector.register("yolov8s",YoloV8DetectorInitiator('yolov8s'))
Detector.register("yolov8x",YoloV8DetectorInitiator('yolov8x'))
Detector.register("yolov8x6",YoloV8DetectorInitiator('yolov8x6'))\

Detector.register("yolov5l6u",YoloV8DetectorInitiator('yolov5l6u'))
Detector.register("yolov5lu",YoloV8DetectorInitiator('yolov5lu'))
Detector.register("yolov5m6u",YoloV8DetectorInitiator('yolov5m6u'))
Detector.register("yolov5mu",YoloV8DetectorInitiator('yolov5mu'))
Detector.register("yolov5n6u",YoloV8DetectorInitiator('yolov5n6u'))
Detector.register("yolov5nu",YoloV8DetectorInitiator('yolov5nu'))
Detector.register("yolov5s6u",YoloV8DetectorInitiator('yolov5s6u'))
Detector.register("yolov5su",YoloV8DetectorInitiator('yolov5su'))
Detector.register("yolov5x6u",YoloV8DetectorInitiator('yolov5x6u'))
Detector.register("yolov5xu",YoloV8DetectorInitiator('yolov5xu'))

Detector.register("yolov3-sspu",YoloV8DetectorInitiator('yolov3-sspu'))
Detector.register("yolov3-tinyu",YoloV8DetectorInitiator('yolov3-tinyu'))
Detector.register("yolov3u",YoloV8DetectorInitiator('yolov3u'))

import torch
import torch.nn as nn

