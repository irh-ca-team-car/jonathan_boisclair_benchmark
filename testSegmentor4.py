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

while True:
    random.shuffle(models)
    for name in tqdm(models,desc="models",leave=False):
        if name in state.processed and state.processed[name] >= state.current_epoch:
            continue
        if name not in state.processed:
            state.processed[name]=0
        repeat = True
        while repeat:
            vsim = subprocess.run(['python3','testSegmentor4Worker.py',name])

            rt = vsim.returncode 
            repeat = rt==5

            print(rt)

    state.current_epoch += 5
    torch.save(state, "a3_weights/state_v.chk")
        
