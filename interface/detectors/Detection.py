from typing import List
import torch
import torch.nn as nn
class Box2d:
    x:float
    y:float
    w:float
    h:float
    c:torch.Tensor
    cf:torch.Tensor
    def __init__(self) -> None:
        self.x=self.y=self.w=self.h=0
        self.c=torch.Tensor()
        self.cf=torch.Tensor()
    def __str__(self) -> str:
        return f"Box2d[x:{self.x},y:{self.y},w:{self.w},h:{self.h},class:{self.c},confidence:{self.cf}]"
class Box3d:
    x:float
    y:float
    z:float
    w:float
    h:float
    d:float
    c:torch.Tensor
    cf:torch.Tensor
    def __init__(self) -> None:
        self.x=self.y=self.w=self.h=self.z=self.d=0
        self.c=torch.Tensor()
        self.cf=torch.Tensor()

    def __str__(self) -> str:
        return f"Box3d[c:{self.x},y:{self.y},z:{self.z},w:{self.w},h:{self.h},d:{self.d},class:{self.c},confidence{self.cf}]"
class Detection:
    boxes2d:List[Box2d]
    boxes3d:List[Box3d]
    def __init__(self) -> None:
        self.boxes2d = []
        self.boxes3d = []
    def fromTorchVision(torchVisionResult):
        print(torchVisionResult)
        ret = []
        for res in torchVisionResult:
            ret.append(Detection())

        if len(ret)==1:
            return ret[0]
        if len(ret)==0:
            return None
        return ret
    def __str__(self) -> str:
        return "Detection [boxes2d:"+str(self.boxes2d)+", boxes3d:"+str(self.boxes3d) + "]"
