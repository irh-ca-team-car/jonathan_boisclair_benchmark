
from typing import List, Optional, Union
from ..datasets import Sample,Size
import torchvision.transforms.functional
import torch
class RequiresGrad():
    def __init__(self,requires_grad:bool):
        self.requires_grad = requires_grad
    def __call__(self,sample: torch.Tensor):
        sample.requires_grad_ = self.requires_grad
        return sample
        