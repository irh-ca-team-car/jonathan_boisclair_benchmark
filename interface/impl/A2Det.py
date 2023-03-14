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

from .a2_det.src.scripts.pytorch_defineNet import *
from .a2_det.src.scripts.pytorch_utils import *
from .a2_det.src.scripts.pytorch_datasets import *
from .a2_det.src.scripts.pytorch_parse_config import *
from .a2_det.src.scripts.pytorch_network import get_module,set_input_size, get_input,get_frequency, get_module_list,set_nb_class,set_nb_chan,set_layer_sep,get_extra, get_layers,get_input_size
from .a2_det.src.scripts.pytorch_network import build_simpler_ng as build_network_simpler
from .a2_det.src.scripts.MultiboxLoss import MultiboxLoss
from .a2_det.src.scripts.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors, convert_locations_to_boxes, convert_boxes_to_locations, center_form_to_corner_form

a2_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "a2_det")


iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(38, 8/300.0, SSDBoxSizes(0.1, 0.2), [2]),
    SSDSpec(19, 16/300.0, SSDBoxSizes(60/300.0, 111/300.0), [2, 3]),
    SSDSpec(10, 32/300.0, SSDBoxSizes(111/300.0, 162/300.0), [2, 3]),
    SSDSpec(5, 64/300.0, SSDBoxSizes(162/300.0, 213/300.0), [2, 3]),
    SSDSpec(3, 100/300.0, SSDBoxSizes(213/300.0, 264/300.0), [2]),
    SSDSpec(1, 300/300.0, SSDBoxSizes(264/300.0, 315/300.0), [2])
]



class A2Det(Detector):
    model: torch.nn.Module
    dataset: DetectionDataset
    isTrain: bool
    size: Size
    device: torch.device
    sep: int
    model_name:str
    nc:int
    criterion:MultiboxLoss

    def __init__(self,sep:int,nc=3,mdl="src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-vgg.cfg",input_size:Size=None) -> None:
        super(A2Det, self).__init__(3, False)
        if input_size is None: input_size = Size(300,300)
        self.size = input_size
        self.device = "cpu"
        self.sep = sep
        self.nc = nc

        self.model_path = mdl if os.path.isabs(mdl) else os.path.join(a2_dir,mdl)
        self.model_name = os.path.basename(self.model_path)
        self.dataset = DetectionDataset.named("coco-empty")
        
        self.isTrain=False

        self.build()
    @property
    def nb_class(self):
        return self.dataset.classesList().__len__()
    def build(self):
        self.model = build_network_simpler(self.model_path, True,sep=self.sep, nb_chan=self.nc, nb_class=len(self.dataset.classesList())+1).to(self.device)
        self.priors = generate_ssd_priors(specs, 1)
        self.priors=self.priors.to(self.device)
        self.target_transform = MatchPrior(self.priors, center_variance, size_variance, 0.5)

    def load_state_dict(self, state_dict, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict)
    def state_dict(self):
        return self.model.state_dict()
    def train(self):
        pass
    def eval(self):
        pass
    def to(self,device:torch.device):
        super(A2Det,self).to(device)
        self.model = self.model.to(device)
        self.priors=self.priors.to(self.device)
        return self
    @torch.inference_mode()
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        rgb = rgb.to(self.device)
        
        is_batch = True
        if (len(rgb.shape) ==3):
            rgb = rgb.unsqueeze(0)
            is_batch=False
        o_shape = rgb.shape
        rgb = torch.nn.functional.interpolate(rgb,size=(300,300)) *255.0
        rst = self.model(rgb)
        confidences,locations=self.extract()
        boxes=convert_locations_to_boxes(locations, self.priors,center_variance,size_variance)#cxcyhw
        boxes=center_form_to_corner_form(boxes)#x1,y1,x2,y2
        boxes[:,:,0:2] -= boxes[:,:,2:]#x1,y1,w,h
        confidences = torch.log_softmax(confidences.float(),2)

        result=[]
        for i in range(confidences.shape[0]):

            argmax = torch.argmax(confidences[i],1) > 0

            argmax_cf=argmax.view([*argmax.shape,1]).repeat(1,self.nb_class+1)
            argmax_loc=argmax.view([*argmax.shape,1]).repeat(1,4)

            #argmax_cf = torch.all(torch.isnan(confidences[i]).logical_not(),1).repeat(1,self.nb_class+1).view(-1,self.nb_class+1)
            #argmax_loc = torch.all(torch.isnan(confidences[i]).logical_not(),1).repeat(1,4).view(-1,4)

            #argmax_cf = torch.BoolTensor(size= [confidences[i].shape])
            #argmax_loc = torch.BoolTensor(size= [boxes[i].shape])

            cf = confidences[i,argmax_cf].view(-1,self.nb_class+1)
            loc = boxes[i,argmax_loc].view(-1,4)

            cf= torch.where(torch.isnan(cf),cf,0)
            loc= torch.where(torch.isnan(loc),loc,0)

            negative_x = torch.logical_and(loc[:,0] > -0.03,loc[:,0] + loc[:,2] < 1.03)
            negative_y = torch.logical_and(loc[:,1] > -0.03,loc[:,1] + loc[:,3] < 1.03)
            mask = torch.logical_and(negative_x,negative_y)
            cf = cf[mask,:]
            loc = loc[mask,:]

            #cf = confidences[i]
            #loc = boxes[i]

            det = Detection()
            for b in range(cf.shape[0]):
                box = Box2d()

                box.c = int(torch.argmax(cf[b,1:]))+1
                box.cf = float(cf[b,box.c-1])

                box.x = float(loc[b,0]*o_shape[3])
                box.y = float(loc[b,1]*o_shape[2])
                box.h = float(loc[b,2]*o_shape[3])
                box.w = float(loc[b,3]*o_shape[2])
                
                if torch.any(torch.isnan(torch.tensor([box.c,box.cf,box.x,box.y,box.w,box.h]))):continue
                
                if box.c ==0:continue
                if box.x < o_shape[3]:continue
                if box.y < o_shape[2]:continue
                if box.w > 2*o_shape[3]:continue
                if box.h > 2*o_shape[2]:continue
                box.cn = self.dataset.getName(box.c)
                det.boxes2d.append(box)

            result.append(det)

        return result if is_batch else result[0]

  
    def adaptTo(self, dataset):
        if self.dataset != dataset:
            
            state_dict = self.model.state_dict()
            self.dataset = dataset
            self.build()
            try:
                self.model.load_state_dict(state_dict,strict=False)
            except:
                pass
            
            return self
        else:
            return self
    def extract(self):
        confidences = []
        locations = []
        end_layers = (self.model.layers)
        if self.sep > 0:
            end_layers = (self.model.layers[-1]).modules
        for m in end_layers:
            if (type(m) is SSDCompute):
                if hasattr(m, "confidence") :
                    confidences.append(m.confidence)
                if hasattr(m, "location"):
                    locations.append(m.location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences,locations
    
 
    def applyTransform(self,x):

        return self.target_transform(x[0],x[1]+1)
    def calculateLoss(self,sample:Sample):
        self.train()
        self.criterion = MultiboxLoss(self.priors, iou_threshold=0.5, neg_pos_ratio=3,
                         center_variance=0.1, size_variance=0.2, device=self.device)
        
        scale = ScaleTransform(Size(300,300))
        sample = scale(sample)
        inp = Cat().__call__(sample).to(self.device)
        rst = self.model(inp)
        target= sample
        if isinstance(target,list):
            target = [t.detection.toX1Y1X2Y2C(self.device) for t in target]
        else:
            target = [target.detection.toX1Y1X2Y2C(self.device)]
        tmp = [(t[:,0:4]/300.0,t[:,4].type(torch.long)) for t in target]
        tmp = [self.applyTransform(x) for x in tmp]

        outputs_locations, outputs_labels= [torch.cat([y[x].reshape(1, *y[x].shape) for y in tmp]) for x in range(len(tmp[0]))]
        boxes=convert_locations_to_boxes(outputs_locations, self.priors,center_variance,size_variance)#cxcyhw

        # print(target[0])
        # print(boxes[outputs_labels>0,:],outputs_labels[outputs_labels>0])
        # exit(0)

        confidence, locations = self.extract()
        #confidence=torch.log_softmax(confidence.float(),2)

        regression_loss, classification_loss= self.criterion(confidence, locations, outputs_labels, outputs_locations)  # TODO CHANGE BOXES
        
        return regression_loss + classification_loss
    @staticmethod
    def optimizer(model):
        return torch.optim.Adamax(model.parameters(),lr=2e-3, weight_decay=1e-6)


class A2DetInitiator():
    def __init__(self,sep:int,nc=3,mdl="src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-vgg.cfg",input_size:Size=None):
        self.sep = sep
        self.nc = nc
        self.mdl = mdl
        self.input_size=input_size
        pass
    def __call__(self):
        return A2Det(self.sep,self.nc,self.mdl,self.input_size)

#for k in range(14):
if True:
    k=8
    m=["src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-vgg.cfg",
    "src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-alexnet.cfg",
    "src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-cae.cfg",
    "src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-densenet.cfg"]
    for path in m:
        basename = os.path.basename(path).replace("model-ssd-","").replace(".cfg","")
        Detector.register("A2_DET_"+basename+"_"+str(k), A2DetInitiator(k,nc=1,mdl=path))