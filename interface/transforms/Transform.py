from typing import List, Union
from ..datasets import Sample
import torch
def apply(samples, transforms:List) -> Union[Sample,List[Sample], torch.Tensor]:
    for x in transforms:
        if isinstance(x,str) or isinstance(x,torch.device):
            if isinstance(samples,list):
                samples = [samp.to(x) for samp in samples]
            else:
                samples = samples.to(x)
        else:
            samples = x(samples)
    return samples