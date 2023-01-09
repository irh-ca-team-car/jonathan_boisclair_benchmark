from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision
from ..datasets.Sample import Sample

class Box2d:
    x: float
    y: float
    w: float
    h: float
    c: float
    cf: float
    cn: str

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = 0
        self.c = 0
        self.cf = 0
        self.cn = ""
    def scale(self,x=1.0,y=1.0):
        newBox = Box2d()
        newBox.x = self.x*x
        newBox.y = self.y*y
        newBox.w = self.w*x
        newBox.h = self.h*y
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        return newBox



    def __str__(self) -> str:
        return f"Box2d[x:{self.x},y:{self.y},w:{self.w},h:{self.h},class:{self.c},confidence:{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


class Box3d:
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float
    c: float
    cf: float
    cn: str
    def scale(self,x=1.0,y=1.0, z=1.0):
        newBox = Box3d()
        newBox.x = self.x*x
        newBox.y = self.y*y
        newBox.z = self.z*z
        newBox.w = self.w*x
        newBox.h = self.h*y
        newBox.d = self.d*y
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        return newBox

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = self.z = self.d = 0
        self.c = 0
        self.cf = 0
        self.cn = ""

    def __str__(self) -> str:
        return f"Box3d[c:{self.x},y:{self.y},z:{self.z},w:{self.w},h:{self.h},d:{self.d},class:{self.c},confidence{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


class Detection:
    boxes2d: List[Box2d]
    boxes3d: List[Box3d]

    def __init__(self) -> None:
        self.boxes2d = []
        self.boxes3d = []
        
    def scale(self,x=1.0,y=1.0):
        newDet = Detection()
        newDet.boxes2d = [b.scale(x,y) for b in self.boxes2d]
        newDet.boxes3d = list([b.scale() for b in self.boxes3d])
        return newDet

    def fromTorchVision(torchVisionResult, dataset=None):
        ret = []
        for res in torchVisionResult:
            det = Detection()
            for b in range(res["boxes"].shape[0]):
                box = Box2d()
                box.x = res["boxes"][b, 0].item()
                box.y = res["boxes"][b, 1].item()
                box.w = res["boxes"][b, 2].item()-res["boxes"][b, 0].item()
                box.h = res["boxes"][b, 3].item()-res["boxes"][b, 1].item()
                box.c = res["labels"][b].item()
                box.cf = res["scores"][b].item()
                box.cn = str(box.c)#CocoDetection.getName(box.c)
                if dataset is not None:
                    box.cn = dataset.getName(box.c)
                det.boxes2d.append(box)
            ret.append(det)

        if len(ret) == 1:
            return ret[0]
        if len(ret) == 0:
            return None
        return ret

    def filter(self, th):
        newVal = Detection()
        newVal.boxes2d = [x for x in self.boxes2d if x.cf > th]
        newVal.boxes3d = [x for x in self.boxes3d if x.cf > th]
        return newVal
    def c(self,c):
        d = Detection()
        d.boxes2d = [x for x in self.boxes2d if int(x.c) == int(c)]
        d.boxes3d = [x for x in self.boxes3d if int(x.c) == int(c)]
        return d
    def onImage(self, sample: Sample, colors:List[Tuple[int,int,int]]=None):
        if isinstance(sample,Sample):
            img = (sample.getRGB()*255.0).byte()
        elif isinstance(sample,torch.Tensor):
            img = sample
        else :
            raise Exception("Argument sample must be sample or tensor")
        target = self.toTorchVisionTarget("cpu")
        if len(self.boxes2d) > 0:
            labels = [b.cn for b in self.boxes2d]
            if colors is not None:
                colors = [c for c in colors]
                i=0
                while len(colors) < len(labels):
                    colors.append(colors[i])
                    i+=1
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=4, colors=colors)
                
                pass
            else:
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=4)
        return img
    def toX1Y1X2Y2C(self,device):
        ret= torch.tensor([[x.x,x.y,x.x+x.w,x.y+x.h,x.c-1]for x in self.boxes2d]).to(device)
        if(len(ret.shape) == 1):
            ret = ret.view(0,5)
        return ret
    def toTorchVisionTarget(self, device):
        boxes = []
        labels = []
        for box in self.boxes2d:
            if box.w<1:
                box.w=2
            if box.h<1:
                box.h=2
            boxes.append([box.x, box.y, box.x+box.w, box.y+box.h])
            if boxes[-1][2] <= boxes[-1][0]:
                boxes[-1][2] = boxes[-1][0] +1
            if boxes[-1][3] <= boxes[-1][1]:
                boxes[-1][3] = boxes[-1][1] +1
            labels.append(int(box.c))
        t= torch.tensor(boxes, dtype=torch.int64).to(device)
        if(len(boxes)==0):
            t = t.view(0,4)
        return {'boxes': t, 'labels': torch.tensor(labels, dtype=torch.int64).to(device)}

    def __str__(self) -> str:
        return "{ type:Detection, boxes2d:"+str(self.boxes2d)+", boxes3d:"+str(self.boxes3d) + "}"

    def __repr__(self) -> str:
        return self.__str__()
