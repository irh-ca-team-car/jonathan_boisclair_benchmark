from typing import List
from .. import Sample
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d
import torchvision.transforms
import os
import torchvision.io

class A1Detection(DetectionDataset):
    A1Classes = ["void"	,"traffic signal","car","bike","pedestrian"]
    def classesList(self):
        return list(A1Detection.A1Classes)
    def getId(self,str:str):
        import sys
        if str in A1Detection.A1Classes:
            return A1Detection.A1Classes.index(str)
        else:
            print(str,"is not a known category from A1",file=sys.stderr)
            
            return A1Detection.getId("void")
    def getName(self,id=None):
        if id is None:
            return "A1"
        if id>=0 and id < len(A1Detection.A1Classes):
            return A1Detection.A1Classes[id]
        return "void"
    def isBanned(self,nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId == "void"
        else:
            return A1Detection.isBanned(A1Detection.getName(nameOrId))

    images: List[str]
    def __init__(self, txtFile:str=None) -> None:
        if txtFile is None:
            return
        import pathlib
        self.root = pathlib.Path(txtFile).parent.parent
        
        self.images = []
        
        f = open(txtFile,"r+")
        lines = f.readlines()
        f.close()
        for line in lines:
            if not self.root.name in line:
                continue
            line = line.strip()
            line = line[line.index(self.root.name)+self.root.name.__len__()+1:]
            self.images.append(os.path.join(self.root,line))
        toRemove=[]
        for path in self.images:
            try:
                #img = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB).float()/255.0
                label = path.replace("/images/","/labels/").replace(".jpg",".txt")
                if not os.path.exists(label):
                    toRemove.append(path)
                    continue
                f= open(label)
                f.close()
                del label
                if not os.path.exists(path):
                    toRemove.append(path)
            except:
                toRemove.append(path)
        for path in toRemove:
            self.images.remove(path)
    def withMax(self,max) -> "A1Detection":
        coco = A1Detection()
        coco.images = self.images[:max]
        return coco
    def withSkip(self,maxValue) -> "A1Detection":
        coco = A1Detection()
        coco.images = self.images[maxValue:]
        return coco
    def shuffled(self) -> "A1Detection":
        import random
        coco = A1Detection()
        coco.images = [x for x in self.images]
        random.shuffle( coco.images )
        return coco

    def __len__(self):
        return len(self.images)
    def label(self,index:int) -> str:
        return self.images[index].replace("/images/","/labels/").replace(".jpg",".txt")
    def __getitem__(self, index: int) -> Sample:
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            return [self.__getitem__(v) for v in values]
        group = self.images[index]

        img = torchvision.io.read_image(group, torchvision.io.ImageReadMode.RGB).float()/255.0
        label = self.label(index)
        det = Detection()
        
        f = open(label,"r+")
        lines = f.readlines()
        f.close()

        for line in lines:
            box = Box2d()
            line = line.strip()
            line = line.split()
            box.c = int(line[0])+1
            box.cn = A1Detection.getName(box.c)
            box.x = float(line[1])* img.shape[2]
            box.y = float(line[2])* img.shape[1]
            box.w = float(line[3])* img.shape[2]
            box.h = float(line[4])* img.shape[1]

            det.boxes2d.append(box)
        
        citiSamp = Sample()
        citiSamp.setImage(img)
        citiSamp.setTarget(det)

        return citiSamp

