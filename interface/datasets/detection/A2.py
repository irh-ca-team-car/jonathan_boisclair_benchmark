from typing import List
from .. import Sample
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d
import torchvision.transforms
import os
import torchvision.io
import numpy as np
import torch

class A2Group:
    def __init__(self):
        self.img = None
        self.label = None
        self.thermal = None
        pass
    def __repr__(self) -> str:
        return self.__dict__.__str__()

class A2Detection(DetectionDataset):
    A2Classes = ["void"	,"traffic signal","car","bike","pedestrian"]
    def classesList(self):
        return list(A2Detection.A2Classes)
    def getId(self,str:str):
        import sys
        if str in A2Detection.A2Classes:
            return A2Detection.A2Classes.index(str)
        else:
            print(str,"is not a known category from A2",file=sys.stderr)
            
            return A2Detection.getId("void")
    def getName(self,id=None):
        if id is None:
            return "A2"
        if id>=0 and id < len(A2Detection.A2Classes):
            return A2Detection.A2Classes[id]
        return "void"
    def isBanned(self,nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId == "void"
        else:
            return A2Detection.isBanned(A2Detection.getName(nameOrId))

    images: List[A2Group]
    def __init__(self, txtFile:str) -> None:
        import pathlib
        self.root = pathlib.Path(txtFile).parent
        
        self.images = []
        
        f = open(txtFile,"r+")
        lines = f.readlines()
        f.close()
        for line in lines:
            group = A2Group()
            line = line.strip().split(",")
            img = line[0][2:]
            thermal = line[1][2:]
            label = line[2]
            if(len(img)>0):
                group.img = os.path.join(self.root,img)
                if not os.path.exists(group.img):
                    print("does not exists:",group.img)
                    continue
            if(len(thermal)>0):
                group.thermal = os.path.join(self.root,thermal)
                if not os.path.exists(group.thermal):
                    print("does not exists:",group.thermal)
                    continue
            if(len(label)>0):
                group.label = os.path.join(self.root,label)
                if not os.path.exists(group.label):
                    print("does not exists:",group.label)
                    continue
            self.images.append(group)
            
            #line = line[line.index(self.root.name)+self.root.name.__len__()+1:]
            #self.images.append(os.path.join(self.root,line))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            return [self.__getitem__(v) for v in values]
        group = self.images[index]
        citiSamp = Sample()
        if group.img is not None:
            img = torchvision.io.read_image(group.img, torchvision.io.ImageReadMode.RGB).float()/255.0
            citiSamp.setImage(img)

        if group.thermal is not None:
            import tifffile as tif
            image = tif.imread(group.thermal)
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / (2**16 -1)
            image = torch.tensor(image).unsqueeze(0)
            citiSamp.setThermal(image)
            pass

        label = group.label
        det = Detection()
        
        f = open(label,"r+")
        lines = f.readlines()
        f.close()

        for line in lines:
            box = Box2d()
            line = line.strip()
            line = line.split()
            box.c = int(line[0])+1
            box.cn = A2Detection.getName(box.c)
            box.x = float(line[1])* img.shape[2]
            box.y = float(line[2])* img.shape[1]
            box.w = float(line[3])* img.shape[2]
            box.h = float(line[4])* img.shape[1]

            det.boxes2d.append(box)
        
        citiSamp.setTarget(det)

        return citiSamp

