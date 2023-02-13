from interface.datasets.detection.Coco import CocoDetection
from interface.detectors import Detector, Detection, Box2d
from interface.detectors import Sample
import interface.detectors
import interface
import torch
import time
import cv2
import torchvision

def datasetAttempt():
    from interface.metrics.Metrics import DatasetAveragePrecision
    dataDir = 'interface/datasets/coco'
    dataType = 'val2014'
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    dataset = CocoDetection("interface/datasets/coco/imgs", annFile).shuffled().withMax(100)

    model = Detector.named("yolov5l").adaptTo(dataset).to("cuda:0")
    model.eval()

    map = DatasetAveragePrecision(model,dataset, verbose = True)
    print(map.mAP(0.000000001))

datasetAttempt()


exit(0)

det = Detection()
box2d = Box2d()
box2d.x=150
box2d.y=340
box2d.w=130
box2d.h=487-340
box2d.c=3
det.boxes2d.append(box2d)
box2d = Box2d()
box2d.x=150
box2d.y=340
box2d.w=130
box2d.h=487-340
box2d.c=4
det.boxes2d.append(box2d)
box2d = Box2d()
box2d.x=150
box2d.y=340
box2d.w=100
box2d.h=100
box2d.c=4
det.boxes2d.append(box2d)

gt = Detection()
box2d = Box2d()
box2d.x=250
box2d.y=340
box2d.w=130
box2d.h=487-340
box2d.c=3
gt.boxes2d.append(box2d)
box2d = Box2d()
box2d.x=150
box2d.y=240
box2d.w=150
box2d.h=487-340
box2d.c=5
gt.boxes2d.append(box2d)
box2d = Box2d()
box2d.x=150
box2d.y=340
box2d.w=150
box2d.h=487-340
box2d.c=7
gt.boxes2d.append(box2d)

from interface.metrics.Metrics import AveragePrecision, MultiImageAveragePrecision

m =AveragePrecision(gt,det)
print(m.precision(0.1))
print(m.recall(0.1))
print(m.coco())
print("-----")
m =AveragePrecision(gt,gt)
print(m.precision(0.5))
print(m.recall(0.5))
print(m.coco())
print("-----")
m= MultiImageAveragePrecision([gt,gt,gt],[det,gt,det])
print(m(0.1))
print(m.coco())
print(m.mAP())
print("-----")
m= MultiImageAveragePrecision([gt],[det])
print(m(0.1))
print(m.coco())
print(m.mAP())