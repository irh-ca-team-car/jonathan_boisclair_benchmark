from dataclasses import dataclass, field
from typing import Dict
from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets import DetectionDataset
from interface.metrics.Metrics import mIOU
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import math
from interface.transforms.Scale import ScaleTransform
import os

#m_name="unet+tu-tinynet_a"
try:
    os.mkdir("a3_weights")
except:
    pass

models= ["unet+mit_b1","unet+mit_b3","unet+mit_b5","fpn+mit_b1","fpn+mit_b3","fpn+mit_b5"]
suffixes = ["_water","","_snow"]
suffixes2 = ["_wr",""] 

@dataclass
class settings():
    processed:Dict = field(default_factory=dict)
    current_epoch:int =10

state = settings()
try:
    state= torch.load("a3_weights/state_v.chk")
except:
    pass

import subprocess

for suffix in tqdm(suffixes,desc="mode",leave=True):
    for suffix2 in tqdm(suffixes2,desc="mode",leave=True):
        for name in tqdm(models,desc="models",leave=True):
            if name in state.processed and state.processed[name] >= state.current_epoch:
                continue
            if name not in state.processed:
                state.processed[name]=0
            repeat = True
            while repeat:
                print("")
                print("")
                print("")
                vsim = subprocess.run(['python3','testSegmentor4Worker.py',name, suffix+suffix2])
                rt = vsim.returncode 
                repeat = rt==5
                print(str(rt))
                print("")

        
