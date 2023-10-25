import sys
from dataclasses import dataclass, field
from typing import Dict
from interface.ITI.impl.CAEbase.weatheradder.OverlayAdder import OverlayAdder
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
import pandas as pd
#dataset = DetectionDataset.named("voc-2007")

dataset = CocoFODetection(split="train", max_samples=20000, type_=["segmentations"], classes=[
                          "car", "truck", "traffic light", "stop sign", "bus", "person", "bicycle", "motorcycle"])
dataset.lazy()

suffix = ""

df = None
def append_to_csv(C_name, image_id, value, save=True):
    global df
    if df is None:
        try:
            df = pd.read_csv("A3"+suffix+".csv")
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
        df.to_csv("A3"+suffix+".csv", index=False)


if(len(dataset) == 0):
    print("reload")
    exit(5)
device = "cuda:0"
water = OverlayAdder("interface/ITI/impl/CAEbase/weatheradder/drop").to(device)
snow = OverlayAdder("interface/ITI/impl/CAEbase/weatheradder/snow").to(device)

# m_name="unet+tu-tinynet_a"
try:
    os.mkdir("a3_weights")
except:
    pass


@dataclass
class settings():
    processed: Dict = field(default_factory=dict)
    current_epoch: int = 10

scale = ScaleTransform(224, 224)
need_exit = False

if len(sys.argv) < 2:
    print("Error, missing arguments ", __file__, "SEGMENTER")
    exit(2)

try:
    model_name = sys.argv[1]
except:
    print("Error,Invalid argument ", __file__, "SEGMENTER")
    exit(2)

try:
    suffix = sys.argv[2]
except:
    suffix = ""

for name in tqdm([model_name], desc=model_name+""+suffix, leave=False):
    model_ctr = Segmenter.named(name).to(device)
    # Segmenter.named("deeplabv3_resnet50")
    model = model_ctr.adaptTo(dataset).to(device)
    try:
        model.load_state_dict(torch.load(
            "a3_weights/"+name+".pth", map_location=device), strict=False)
    except:
        pass
   
    b_size = 1
    batch = Batch.of(dataset, b_size)

    iter = 0
    mIou = None
    t = tqdm(batch, desc="batch", leave=False)
    acc = mIOUAccumulator(len(dataset.classesList()))

    for cocoSamp in t:
        cocoSamp = scale(cocoSamp)

        for samp in cocoSamp:
            samp = samp.to(device)
            if suffix == "_water":
                water.add(samp.getRGB(),200,0.2)
            if suffix == "_snow":
                snow.add(samp.getRGB(),200,0.2)
                snow.add(samp.getRGB(),75,0.5)
                snow.add(samp.getRGB(),10,1)

        prediction = [f.filter(0.8) for f in model.forward(cocoSamp)]
        mIou = mIOU([x.segmentation for x in cocoSamp], prediction)

        append_to_csv(model_name, iter, mIou.calc(),False)
        iter += len(cocoSamp)
        if iter % 100 ==0:
            df = df.sort_values(by="Image")
            df.to_csv("A3"+suffix+".csv", index=False)
        if need_exit:
            break
    df = df.sort_values(by="Image")
    df.to_csv("A3"+suffix+".csv", index=False)

    if need_exit:
        exit(0)
        break
exit(0)
