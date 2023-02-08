from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.detection.Coco import CocoDetection
from interface.datasets.Batch import Batch
from interface.transforms import RandomCutTransform, RandomRotateTransform, rotate, AutoContrast
from interface.transforms import ScaleTransform
import interface.transforms
import interface
import torch
import time
import cv2
import torchvision

sample = Sample.Example()
sample.detection = Detection()
sample.detection.boxes2d.append(Box2d())
sample.detection.boxes2d[0].x = 150
sample.detection.boxes2d[0].y = 325
sample.detection.boxes2d[0].w = 125
sample.detection.boxes2d[0].h = 175
sample.detection.boxes2d[0].c = 1
sample.detection.boxes2d[0].cn = "car"

autoContrast = AutoContrast()

rgb = sample.detection.onImage(sample)

sample.show(rgb,True)

randomCrop = RandomCutTransform(100,100,0.2,True)
transform2 = ScaleTransform(640,640)
rotation = RandomRotateTransform([x for x in range(360)])

for i in range(100):
    sample2 = interface.transforms.apply(sample,[rotation,randomCrop, transform2, autoContrast])
    rgb = sample2.detection.onImage(sample2)
    sample.show(rgb,True)
