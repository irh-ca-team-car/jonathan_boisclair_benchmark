import sys
try:
    import fiftyone
    fiftyone.core.logging.set_logging_level(40)
except:
    pass
from dataclasses import dataclass, field
from typing import Dict, List
from interface.ITI import ITI
from interface.ITI.impl.CAEbase.weatheradder.OverlayAdder import OverlayAdder
from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample, Segmentation
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
dfe = None
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
def append_to_csv_E(C_name, image_id, value, gt, save=True):
    global dfe
    if gt > 0.01:
        if dfe is None:
            try:
                dfe = pd.read_csv("A3E"+suffix+".csv")
            except:
                dfe = pd.DataFrame([], columns=['Image', 'gt'])
        if not C_name in dfe.columns:
            dfe[C_name] = None
        row = dfe[dfe["Image"] == image_id]
        if(len(row) == 0):
            df2 = pd.DataFrame([[image_id, value, gt]], columns=['Image', C_name, 'gt'])
            dfe = pd.concat([dfe, df2])
        else:
            dfe.loc[dfe["Image"] == image_id, C_name] = value

        if save:
            dfe = dfe.sort_values(by="Image")
            dfe.to_csv("A3E"+suffix+".csv", index=False)

if(len(dataset) == 0):
    print("reload")
    exit(5)
device = "cuda:0"
water = OverlayAdder("interface/ITI/impl/CAEbase/weatheradder/drop").to(device)
snow = OverlayAdder("interface/ITI/impl/CAEbase/weatheradder/snow").to(device)
wr_name="unet+++resnet18_3->3"
wr_model = ITI.named(wr_name)().to(device)
try:
    wr_model.load_state_dict(torch.load(wr_name+".pth", map_location=device), strict=False)
except:
    pass
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
def to_str( *q):
    return " ".join([str(v)for v in q])
with torch.no_grad():
    for name in tqdm([model_name], desc=model_name+""+suffix, leave=False):
        model_ctr = Segmenter.named(name).to(device)
        # Segmenter.named("deeplabv3_resnet50")
        model = model_ctr.adaptTo(dataset).to(device)
        try:
            model.load_state_dict(torch.load(
                "a3_weights/"+name+".pth", map_location=device), strict=False)
        except:
            pass
    
        b_size = 64
        batch = Batch.of(dataset, b_size)

        iter = 0
        mIou = None
        t = tqdm(batch, desc="batch", leave=False)

        for cocoSamp in t:
            cocoSamp :List[Sample] = scale(cocoSamp)

            for samp in cocoSamp:
                samp = samp.to(device)
                if "_water" in suffix :
                    water.add(samp.getRGB(),200,0.2)
                if "_snow" in suffix :
                    snow.add(samp.getRGB(),200,0.2)
                    snow.add(samp.getRGB(),75,0.5)
                    snow.add(samp.getRGB(),10,1)
                if "_wr" in suffix:
                    samp.setImage(wr_model.forward(samp).getRGB())
            
            prediction:List[Segmentation] = [f.filter(0.8) for f in model.forward(cocoSamp)]
            for i in range(len(cocoSamp)):
                mIou = mIOU([cocoSamp[i].segmentation], [prediction[i]])
                append_to_csv(model_name, iter, mIou.calc(),False)
                iter += 1
                if iter % 100 ==0:
                    df = df.sort_values(by="Image")
                    df.to_csv("A3"+suffix+".csv", index=False)
                #energy part
                p = prediction[i].clone()
                p.detection.boxes2d = [x for x in p.detection.boxes2d ]
                gt = cocoSamp[i].clone()
                gt.detection.boxes2d = [x for x in gt.detection.boxes2d ]
                d_max = 0
                d_gt =0
                if len(gt.detection.boxes2d) >0 and len(p.detection.boxes2d):
                    d3 = p.detection.BoxIOU(gt.detection)
                    mask = d3.max(1).values > 0.001
                    d4=d3[mask]
                    boxes = (p.detection.Box2dMask(mask))
                    for f in boxes:
                        d =f.EvaluateDistance(gt)
                        d_max = max(d,d_max)
                for f in gt.detection.boxes2d:
                    d =f.EvaluateDistance(gt)
                    d_gt = max(d,d_gt)
                if d_max > d_gt:
                    d_max = d_gt
                #tqdm.write(to_str(model_name, iter, d_max, d_gt))
                append_to_csv_E(model_name, iter, d_max, d_gt)
                
                
            if need_exit:
                break
        df = df.sort_values(by="Image")
        df.to_csv("A3"+suffix+".csv", index=False)

        if need_exit:
            exit(0)
            break
exit(0)
