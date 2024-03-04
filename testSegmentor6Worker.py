from dataclasses import dataclass, field
from typing import Dict, List
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

model_name = "vit_b_16"
model_name = "alexnet"
df = None
P = "_water"

def append_to_csv(C_name, image_id, value, save=True):
    global df
    if df is None:
        try:
            df = pd.read_csv("A3_V"+P+model_name+".csv")
        except:
            df = pd.DataFrame([], columns=['Image'])
    if not C_name in df.columns:
        df[C_name] = None
    row = df[df["Image"] == image_id]
    if(len(row) == 0):
        df2 = pd.DataFrame([[image_id, value]], columns=['Image', C_name])
        df = pd.concat([df, df2])
    else:
        df.loc[df["Image"] == image_id, C_name] = value

    if save:
        df = df.sort_values(by="Image")
        df.to_csv("A3_V"+P+model_name+".csv", index=False)

if(len(dataset)==0):
    print("reload")
    exit(5)
device = "cuda:0"


c = ClassificationDataset()
df2 = pd.read_csv("A3"+P+".csv")
t_dataset = torch.tensor(df2.values)
t_dataset = t_dataset[:,1:]
c.clz = list(range(t_dataset.shape[1]))
t_dataset =t_dataset.float().to(device)

import sys

#m_name="unet+tu-tinynet_a"
try:
    os.mkdir("a3_weights")
except:
    pass

scale = ScaleTransform(224,224)
need_exit = False


for name in tqdm([model_name],desc="models",leave=False):
    
    model_ctr = Classifier.named(name).to(device)
    model = model_ctr.adaptTo(c).to(device)
    try:
        model.load_state_dict(torch.load("a3_weights/meta"+model_name+".pth", map_location=device), strict=False)
    except:
        pass
    # Assuming optimizer has two groups.

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

        class_ :List[Classification]= model.forward(cocoSamp)

        for i in range(len(cocoSamp)):
            mIou = gt[i][class_[i].getCategory()]
            append_to_csv(model_name, iter+i, float(mIou),False)
            mx = gt[i].max()
            append_to_csv("max", iter+i, float(mx),False)
            pot = mIou / mx
            append_to_csv("potential", iter+i, float(pot),True)

        iter+= len(cocoSamp)
        if need_exit:
            break

    if need_exit:
        exit(0)
        break
  
exit(0)