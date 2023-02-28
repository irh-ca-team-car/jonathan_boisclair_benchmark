
from typing import List, Optional, Union
from ..datasets import Sample,Size
from .Scale import scale
import random
import torch.nn

def FLIR_FIX(sample:Sample):
    x,y,w,h=37, 44, 449, 490

    if isinstance(sample,list):
        return [FLIR_FIX(x) for x in sample]
    sample = scale(sample,400,400)
    rgb = sample.getRGB()
    tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
    sample.setImage(tmp)
    return sample