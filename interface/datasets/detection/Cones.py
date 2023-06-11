from typing import List
from .. import Sample
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d
import torchvision.transforms
import os
import torchvision.io
import numpy as np
import torch
import json


class ConesGroup:
    def __init__(self):
        self.img = None
        self.label = None
        self.thermal = None
        pass

    def __repr__(self) -> str:
        return self.__dict__.__str__()


class ConesDetection(DetectionDataset):
    ConesClasses = ["void", "cone"]

    @staticmethod
    def classesList():
        return list(ConesDetection.ConesClasses)

    @staticmethod
    def getId(str: str):
        import sys
        if str in ConesDetection.ConesClasses:
            return ConesDetection.ConesClasses.index(str)
        else:
            print(str, "is not a known category from ConesDetection", file=sys.stderr)

            return ConesDetection.getId("void")

    @staticmethod
    def getName(id=None):
        if id is None:
            return "ConesDetection"
        if id >= 0 and id < len(ConesDetection.ConesClasses):
            return ConesDetection.ConesClasses[id]
        return "void"

    @staticmethod
    def isBanned(nameOrId):
        if isinstance(nameOrId, str):
            return nameOrId == "void"
        else:
            return ConesDetection.isBanned(ConesDetection.getName(nameOrId))

    images: List[ConesGroup]

    def __init__(self, txtFile: str = None) -> None:
        if txtFile is None:
            self.images = []
            self.min = 0
            self.max = None
            return
        import pathlib
        self.root = pathlib.Path(txtFile).parent
        self.min = 0
        self.max = None
        self.images = []

        for (dirpath, dirnames, filenames) in os.walk(txtFile):
            for filename in filenames:
                if filename.endswith('.json'):
                    group = ConesGroup()
                    group.label = os.path.join(dirpath, filename)
                    group.img = os.path.join(
                        dirpath, filename.replace(".json", ".png"))
                    self.images.append(group)

    def withMax(self, max) -> "ConesDetection":
        coco = ConesDetection()
        coco.images = self.images[:max]
        return coco

    def withSkip(self, maxValue) -> "ConesDetection":
        coco = ConesDetection()
        coco.images = self.images[maxValue:]
        return coco

    def shuffled(self) -> "ConesDetection":
        import random
        coco = ConesDetection()
        coco.images = [x for x in self.images]
        random.shuffle(coco.images)
        return coco

    def __len__(self):
        if self.images is not None:
            if self.max is not None:
                return min(len(self.images)-self.min, self.max)
            return len(self.images)
        return 0

    def __getitem__(self, index: int) -> Sample:
        if isinstance(index, slice):
            values = []
            if index.step is not None:
                values = [v for v in range(
                    index.start, index.stop, index.step)]
            else:
                values = [v for v in range(index.start, index.stop)]
            return [self.__getitem__(v) for v in values]
        group = self.images[index]
        citiSamp = Sample()
        if group.img is not None:
            img = torchvision.io.read_image(
                group.img, torchvision.io.ImageReadMode.RGB).float()/255.0
            citiSamp.setImage(img)

        label = group.label
        det = Detection()

        f = open(label, "r+")
        lines = json.load(f)
        f.close()

        for line in lines:
            file_box = line["rectMask"]
            box = Box2d()
            box.c = 1
            box.cn = "cone"
            box.x = float(file_box["xMin"])
            box.y = float(file_box["yMin"])
            box.w = float(file_box["width"])
            box.h = float(file_box["height"])

            det.boxes2d.append(box)

        citiSamp.setTarget(det)

        return citiSamp
