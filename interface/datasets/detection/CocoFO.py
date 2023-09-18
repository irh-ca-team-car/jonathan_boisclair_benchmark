from typing import List, Literal, Union
from .. import Sample, LidarSample, Segmentation
from ...detectors.Detection import Detection, Box2d, Box3d
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
import open3d as o3d


class CocoFODetection:
    A1Classes = None
    images: fo.Dataset
    data: List[fo.Sample]

    def withMax(self, max) -> "CocoFODetection":
        coco = CocoFODetection()
        coco.data = self.data[:max]
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco

    def withSkip(self, maxValue) -> "CocoFODetection":
        coco = CocoFODetection()
        coco.data = self.data[maxValue:]
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco

    def shuffled(self) -> "CocoFODetection":
        import random
        coco = CocoFODetection()
        coco.data = [x for x in self.data]
        random.shuffle(coco.data)
        self.lazy()
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco

    def lazy(self):
        if not hasattr(self, "images"):
            self.images = foz.load_zoo_dataset(
                "sama-coco", split=self.split, label_types=self.type, **self.kwargs)
            self.data = list(self.images)
        return self

    def __init__(self, split: Union[Literal["train"], Literal["test"], Literal["validation"], None] = "train", dataset_dir=None, type_: Union[Literal["detections"], Literal["segmentations"]] = "detections", **kwargs) -> None:
        kwargs["dataset_dir"] = dataset_dir
        self.split = split
        self.kwargs = kwargs
        self.type = type_

        self.dataset_dir = dataset_dir
        # The directory containing the dataset to import
        self.dataset_dir = dataset_dir

        # Import the dataset

        # print(dataset.group_media_types)
        # for group in (dataset.iter_groups()):
        #    print(group)
        # exit(0)

        self.n = None
        self.split = split
        self.A1Classes = None
        self.data = None

    def classesList(self):
        if self.A1Classes is None:
            #self.A1Classes =["void", *self.images.get_classes("ground_truth")]
            self.lazy()
            if "classes" in self.kwargs:
                self.A1Classes = ["void", *self.kwargs["classes"]]
            else:
                self.A1Classes = ["void", *self.images.distinct(
                    "ground_truth.detections.label"
                )]
        return list(self.A1Classes)

    def getId(self, str: str):
        import sys
        if self.A1Classes is None:
            self.classesList()
        if str in self.A1Classes:
            return self.A1Classes.index(str)
        else:
            return self.getId("void")

    def getName(self, id=None):
        if self.A1Classes is None:
            self.classesList()
        if id is None:
            return "COCO-2017"
        if id >= 0 and id < len(self.A1Classes):
            return self.A1Classes[id]
        return "void"

    def isBanned(self, nameOrId):
        if self.A1Classes is None:
            self.classesList()
        if isinstance(nameOrId, str):
            return nameOrId == "void" or nameOrId == "DontCare"
        else:
            return self.isBanned(self.getName(nameOrId))

    def __len__(self):
        return len(self.lazy().data)

    def __getitem__(self, index: int) -> Sample:
        if isinstance(index, slice):
            values = []
            if index.step is not None:
                values = [v for v in range(
                    index.start, index.stop, index.step)]
            else:
                values = [v for v in range(index.start, index.stop)]
            self.lazy()
            values = [v for v in values if v < len(self.data)]
            if len(values) == 0:
                raise StopIteration
            return [self.__getitem__(v) for v in values]
        else:
            self.lazy()
            value = self.data[index]

            citiSamp = Sample.fromFiftyOne(value)
            sz = citiSamp.size()
            seg_img = torch.zeros((sz.h, sz.w), dtype=torch.float32)
            dict_ = value.to_dict()
            citiSamp.detection = Detection()
            if "ground_truth" in dict_:
                if dict_["ground_truth"] is not None and "detections" in dict_["ground_truth"]:
                    for d in value["ground_truth"]["detections"]:
                        box = Box2d()
                        box.x = int(d["bounding_box"][0] *
                                    citiSamp.getRGB().shape[2])
                        box.y = int(d["bounding_box"][1] *
                                    citiSamp.getRGB().shape[1])
                        box.w = int(d["bounding_box"][2] *
                                    citiSamp.getRGB().shape[2])
                        box.h = int(d["bounding_box"][3] *
                                    citiSamp.getRGB().shape[1])
                        box.c = self.getId(d["label"])
                        box.cn = d["label"]
                        if "classes" in self.kwargs:
                            if box.cn not in self.kwargs["classes"]:
                                continue
                        if not self.isBanned(d["label"]):
                            mask = torch.tensor(d["mask"])
                            citiSamp.detection.boxes2d.append(box)
                            seg_shape = (mask.shape)
                            v = seg_img[box.y:box.y+seg_shape[0],
                                        box.x:box.x+seg_shape[1]]
                            v[mask] = box.c
            citiSamp.segmentation = Segmentation.FromImage(seg_img.float(), None)
            return citiSamp
