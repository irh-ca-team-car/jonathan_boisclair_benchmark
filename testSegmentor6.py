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

import subprocess
suffixes = ["","_snow","_water"]

for suffix in tqdm(suffixes,desc="models",leave=False):

    repeat = True
    while repeat:
        vsim = subprocess.run(['python3','testSegmentor6Worker.py', suffix])

        rt = vsim.returncode 
        repeat = rt==5
        print(rt)

        
