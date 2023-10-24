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

@dataclass
class settings():
    processed:Dict = field(default_factory=dict)
    current_epoch:int =1

state = settings()
try:
    state= torch.load("a3_weights/state_c.chk")
except:
    pass

import subprocess

while True:
    repeat = True
    while repeat:
        torch.save(state, "a3_weights/state_c.chk")
        vsim = subprocess.run(['python3','testSegmentor5Worker.py'])
        state= torch.load("a3_weights/state_c.chk")

        rt = vsim.returncode 
        repeat = rt==5

        print(rt)

    state.current_epoch += 5
    torch.save(state, "a3_weights/state_c.chk")
        
