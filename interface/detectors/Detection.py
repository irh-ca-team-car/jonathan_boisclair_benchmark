from typing import List
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

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = self.z = self.d = 0
        self.c = 0
        self.cf = 0
        self.cn = ""

    def __str__(self) -> str:
        return f"Box3d[c:{self.x},y:{self.y},z:{self.z},w:{self.w},h:{self.h},d:{self.d},class:{self.c},confidence{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


mscoco = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

class Detection:
    boxes2d: List[Box2d]
    boxes3d: List[Box3d]

    def __init__(self) -> None:
        self.boxes2d = []
        self.boxes3d = []
        

    def fromTorchVision(torchVisionResult):
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
                if box.c in mscoco:
                    box.cn = mscoco[box.c]
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

    def onImage(self, sample: Sample):
        img = (sample.getRGB()*255.0).byte()
        target = self.toTorchVisionTarget("cpu")
        if len(self.boxes2d) > 0:
            labels = [b.cn for b in self.boxes2d]
            img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=4)
        return img

    def toTorchVisionTarget(self, device):
        boxes = []
        labels = []
        for box in self.boxes2d:
            boxes.append([box.x, box.y, box.x+box.w, box.y+box.h])
            labels.append(int(box.c))
        return {'boxes': torch.tensor(boxes, dtype=torch.int64).to(device), 'labels': torch.tensor(labels, dtype=torch.int64).to(device)}

    def __str__(self) -> str:
        return "{ type:Detection, boxes2d:"+str(self.boxes2d)+", boxes3d:"+str(self.boxes3d) + "}"

    def __repr__(self) -> str:
        return self.__str__()
