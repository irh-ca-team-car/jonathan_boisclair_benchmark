import fiftyone as fo
import fiftyone.zoo as foz
from dataclasses import dataclass, field
from typing import Dict
from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets import DetectionDataset
from interface.datasets.detection.CocoFO import CocoFODetection
from interface.metrics.Metrics import mIOU, mIOUAccumulator
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import math
from interface.transforms.Scale import ScaleTransform
import os
#dataset = DetectionDataset.named("voc-2007")

dataset = CocoFODetection(split="train",max_samples=10000, type_=["segmentations"], classes=["car","truck","traffic light","stop sign","bus","person","bicycle","motorcycle"])
dataset.lazy()

session = fo.launch_app(dataset.images)

input()