from dataclasses import dataclass, field
from typing import Dict
from interface.segmentation.Segmenter import Segmenter
from interface.classifiers.Classifier import Classifier
from interface.datasets.Sample import Classification, Sample
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
from interface.datasets.classification.ClassificationDataset import ClassificationDataset
import pandas as pd
#dataset = DetectionDataset.named("voc-2007")

dataset = CocoFODetection(split="train",max_samples=10000, type_=["segmentations"], classes=["car","truck","traffic light","stop sign","bus","person","bicycle","motorcycle"])
dataset.lazy()


if(len(dataset)==0):
    print("reload")
    exit(5)
device = "cuda:0"


c = ClassificationDataset()
df = pd.read_csv("A3.csv")
t_dataset = torch.tensor(df.values)
t_dataset = t_dataset[:,1:]
c.clz = list(range(t_dataset.shape[1]))
t_dataset =t_dataset.float().to(device)

import sys

#m_name="unet+tu-tinynet_a"
try:
    os.mkdir("a3_weights")
except:
    pass

@dataclass
class settings():
    processed:Dict = field(default_factory=dict)
    current_epoch:int =10

state = settings()
try:
    state= torch.load("a3_weights/state_c.chk")
except:
    pass

scale = ScaleTransform(224,224)
need_exit = False

model_name = "vit_b_16"
model_name = "alexnet"

for name in tqdm([model_name],desc="models",leave=False):
        if name in state.processed and state.processed[name] >= state.current_epoch:
            print("Already over")
            continue
        if name not in state.processed:
            state.processed[name]=0

        model_ctr = Classifier.named(name).to(device)
        model = model_ctr.adaptTo(c).to(device)
        try:
            model.load_state_dict(torch.load("a3_weights/meta"+model_name+".pth", map_location=device), strict=False)
        except:
            pass
        optim: torch.optim.Optimizer = torch.optim.Adamax(model.parameters()) 
        # Assuming optimizer has two groups.

        for i in tqdm(range(state.processed[name],state.current_epoch),desc="epoch",leave=False):
            b_size = 16
            batch=Batch.of(dataset,b_size)
            batch_gt=Batch.of(t_dataset,b_size)
            
            iter=0
            mIou = None
            t=tqdm(zip(batch, batch_gt),total=len(batch),desc="batch",leave=False)
            
            for cocoSamp, gt in t:
                cocoSamp = scale(cocoSamp)
               
                for i in range(len(cocoSamp)):
                    cocoSamp[i].classification = Classification(gt[i])

                optim.zero_grad()
                loss = (model.calculateLoss(cocoSamp)) / len(cocoSamp)
                loss.backward()

                optim.step()
                optim.zero_grad()

                iter+= len(cocoSamp)
                if need_exit:
                    break

            state.processed[name] += 1
            torch.save(state, "a3_weights/state_c.chk")
            torch.save(model.state_dict(), "a3_weights/meta"+model_name+".pth")
            if need_exit:
                exit(0)
                break
        if need_exit:
            exit(0)
            break
exit(0)