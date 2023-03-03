from typing import List
from .. import Sample
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d
import torchvision.transforms
import os
import torchvision.io
import numpy as np
import torch
from .A2 import A2Detection


class A2W(DetectionDataset):
    A2Classes = ["car"]
    @staticmethod
    def classesList():
        return list(A2W.A2Classes)
    @staticmethod
    def getId(str:str):
        import sys
        if str in A2W.A2Classes:
            return A2W.A2Classes.index(str)
        else:
            #print(str,"is not a known category from A2",file=sys.stderr)
            return A2W.getId("void")
    @staticmethod
    def getName(id=None):
        if id is None:
            return "A2"
        if id>=0 and id < len(A2W.A2Classes):
            return A2W.A2Classes[id]
        return "void"
    @staticmethod
    def isBanned(nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId != "car"
        else:
            return A2W.isBanned(A2W.getName(nameOrId))

    a2: A2Detection
    def __init__(self, a2) -> None:
        self.a2 = a2

    def withMax(self,max) -> "A2Detection":
        self.a2 = self.a2.withMax(max)
        return self
    def withSkip(self,maxValue) -> "A2Detection":
        self.a2 = self.a2.withSkip(maxValue)
        return self
    def shuffled(self) -> "A2Detection":
        self.a2 = self.a2.shuffled()
        return self

    def __len__(self):
        return self.a2.__len__()

    def __getitem__(self, index: int) -> Sample:
        citiSamp = self.a2.__getitem__(index)
        def convert(samp:Sample):
            if samp.detection is not None:
                for box in list(samp.detection.boxes2d):
                    if A2W.isBanned(box.cn):
                        samp.detection.boxes2d.remove(box)
                    else:
                        box.c = A2W.getId(box.cn)
        if isinstance(citiSamp,Sample):
            convert(citiSamp)
        else:
            for samp in citiSamp:
                convert(samp)
        return citiSamp

