from typing import List, Literal, Union
from .. import Sample, LidarSample
from ...detectors.Detection import Detection, Box2d, Box3d
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
import open3d as o3d

class VoC2007Detection:
    A1Classes = None
    images: fo.Dataset
    data : List[fo.Sample]

    def withMax(self,max) -> "VoC2007Detection":
        coco = VoC2007Detection()
        coco.data = self.data[:max]
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco
    def withSkip(self,maxValue) -> "VoC2007Detection":
        coco = VoC2007Detection()
        coco.data = self.data[maxValue:]
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco
    def shuffled(self) -> "VoC2007Detection":
        import random
        coco = VoC2007Detection()
        coco.data = [x for x in self.data]
        random.shuffle( coco.data )
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco

    def lazy(self):
        if not hasattr(self,"images"):
            self.images = foz.load_zoo_dataset("voc-2007", split=self.split, **self.kwargs)

    def __init__(self, split: Union[Literal["train"],Literal["test"], None]=None,dataset_dir=None, **kwargs) -> None:
        kwargs["dataset_dir"]=dataset_dir
        self.split = split
        self.kwargs = kwargs
        #self.images = foz.load_zoo_dataset("voc-2007", split=split, **kwargs)
        self.dataset_dir=dataset_dir
        # The directory containing the dataset to import
        self.dataset_dir =dataset_dir

        # Import the dataset

        #print(dataset.group_media_types)
        #for group in (dataset.iter_groups()):
        #    print(group)
        #exit(0)

        self.n = None
        self.split = split
        self.A1Classes = None
        self.data = None

    def classesList(self):
        if self.A1Classes is None:
            #self.A1Classes =["void", *self.images.get_classes("ground_truth")]
            self.lazy()
            self.A1Classes = ["void",*self.images.distinct(
                "ground_truth.detections.label"
            )]
        return list(self.A1Classes)
    def lazy(self):
        if self.data is None:
            self.lazy()
            self.data = list(self.images)
        return self
    def getId(self,str: str):
        import sys
        if self.A1Classes is None:
            self.classesList()
        if str in self.A1Classes:
            return self.A1Classes.index(str)
        else:
            #print(str, "is not a known category from OpenImages", file=sys.stderr)
            return self.getId("void")

    def getName(self,id=None):
        if self.A1Classes is None:
            self.classesList()
        if id is None:
            return "VoC-2007"
        if id >= 0 and id < len(self.A1Classes):
            return self.A1Classes[id]
        return "void"

    def isBanned(self,nameOrId):
        if self.A1Classes is None:
            self.classesList()
        if isinstance(nameOrId, str):
            return nameOrId == "void" or nameOrId == "DontCare"
        else:
            return self.isBanned(self.getName(nameOrId))

   
    def __len__(self):
        return len(self.lazy().data)

    def __getitem__(self, index: int) -> Sample:

        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            self.lazy()
            values = [v for v in values if v < len(self.images)]
            if len(values)==0:
                raise StopIteration
            return [self.__getitem__(v) for v in values]
        else:
            value = self.lazy().data[index]

            citiSamp = Sample.fromFiftyOne(value)
            
            dict = value.to_dict()

            citiSamp.detection = Detection()
            if "ground_truth" in dict:
                for d in dict["ground_truth"]["detections"]:
                    box = Box2d()
                    box.x = d["bounding_box"][0] * citiSamp.getRGB().shape[2]
                    box.y = d["bounding_box"][1] * citiSamp.getRGB().shape[1]
                    box.w = d["bounding_box"][2] * citiSamp.getRGB().shape[2]
                    box.h = d["bounding_box"][3] * citiSamp.getRGB().shape[1]
                    box.c = self.getId(d["label"])
                    box.cn = d["label"]
                    if not self.isBanned(d["label"]):
                        citiSamp.detection.boxes2d.append(box)
            import pathlib
            
            return citiSamp

