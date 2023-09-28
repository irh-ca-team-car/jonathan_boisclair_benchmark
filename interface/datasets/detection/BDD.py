from typing import Any, List, Tuple
from .. import Sample
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d,Box3d,Segmentation
import torchvision.transforms
import os
import json
import torchvision.io
import numpy as np
import torch
class BDDGroup:
    def __init__(self):
        self.label = None
        self.img = None
        self.id = ""
        pass

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

class BDDDetection(DetectionDataset):
    BDDClasses:List[str]
#      = ["void"	,"road",
# "sidewalk","parking","rail track",
# "human"	,"person","rider",
# "vehicle"	,"car" , "truck" , "bus" , "on rails" , "motorcycle" , "bicycle" , "caravan" , "trailer",
# "construction",	"building" , "wall" , "fence" , "guard rail" , "bridge" , "tunnel",
# "object"	,"pole" , "polegroup" , "traffic sign" , "traffic light",
# "nature",	"vegetation" , "terrain",
# "sky"	,"sky",
# "ground" , "dynamic", "static", "ego vehicle","out of roi","license plate","rectification border"
# ]
    NoTrainClass = ["void"]
    def classesList(self):
        if hasattr(self, "BDDClasses"):
            return list(self.BDDClasses)
        else:
            self.BDDClasses = ["void","drivable"]
            return list(self.BDDClasses)
    def getId(self,str:str):
        import sys
        if str == "train":
            return self.getId("on rails")
        if str in self.BDDClasses:
            return self.BDDClasses.index(str)
        else:
            if "group" in str:
                return self.getId(str.replace("group",""))
            print(str,"is not a known category from citiscapes",file=sys.stderr)
            
            return self.getId("void")
    def getName(self,id=None):
        if id is None:
            return "Citiscapes"
        if id>=0 and id < len(self.BDDClasses):
            return self.BDDClasses[id]
        return "void"
    def isBanned(self,nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId in self.NoTrainClass
        else:
            return self.isBanned(self.getName(nameOrId))

    def withMax(self,max) -> "BDDDetection":
        coco = BDDDetection()
        coco.images = self.images[:max]
        return coco
    def withSkip(self,maxValue) -> "BDDDetection":
        coco = BDDDetection()
        coco.images = self.images[maxValue:]
        return coco
    def shuffled(self) -> "BDDDetection":
        import random
        coco = BDDDetection()
        coco.images = [x for x in self.images]
        random.shuffle( coco.images )
        return coco

    images: List[BDDGroup]
    def __init__(self, root=None, split="train", mode="drivable_maps", classes=2) -> None:
        self.images = []
        images=dict()
        imagesFiles: List[(str,str)] = []
        if classes ==2:
            self.BDDClasses = ["void","drivable"]
        else:
            self.BDDClasses = ["void","drivable","other_lane"]

        if root is None:
            root = "/mnt/sd/datasets/bdd"
        try:
            for (dirpath, dirnames, filenames) in os.walk(root):
                for filename in filenames:
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        if "/"+split in dirpath:
                            imagesFiles.append((dirpath, filename))
        except:
            pass
        for dirpath, filename in imagesFiles:
            id = os.path.splitext(filename)[0][0:17]
            if id in images:
                obj = images[id]
            else:
                obj = images[id] = BDDGroup()
            obj.id = id

            if mode+"/labels" in dirpath:
                obj.label = os.path.join(dirpath, filename)
            elif "images/" in dirpath:
                obj.img = os.path.join(dirpath, filename)

        toRemove = []
        for value in images.values():
            if value.label is None or value.img is None:
                toRemove.append(value.id)

        for key in toRemove:
            del images[key]
        #print(self.images)
        self.images= list(images.values())
        pass
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

        img = torchvision.io.read_image(group.img, torchvision.io.ImageReadMode.RGB).float()/255.0
        img_lbl = torchvision.io.read_image(group.label, torchvision.io.ImageReadMode.GRAY).long()
        if len(self.BDDClasses) <=2:
            img_lbl[img_lbl[:,:,0]==2] =1
        citiSamp = Sample()
        citiSamp.setImage(img)
        citiSamp.segmentation = Segmentation.FromImage(img_lbl, self.classesList())

        return citiSamp

