import fiftyone
import fiftyone.zoo
from interface.datasets import Sample
from interface.datasets.detection import A2Detection
from interface.impl.YoloV7 import YoloV7Detector
from interface.detectors import Detector
import torch
import cv2

sample = Sample.Example()

#detect
workImage = sample.detection.onImage(sample, colors=[(128, 128, 255)])
Sample.show(workImage,True)