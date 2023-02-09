

from typing import List, Optional, Union
from ..datasets import Sample,Size
import random

def random_cut(sample: Union[Sample,List[Sample]],min_width:int,min_height:int, overlap_to_keep=0.2, need_box = True,keep_aspect=False) -> Union[Sample,List[Sample]]:
    def scaleValue(sample:Sample):
        new_width=random.randint(min_width,sample.size().w)
        new_height=random.randint(min_height,sample.size().h)
        new_x  = random.randint(0,sample.size().w - new_width)
        new_y  = random.randint(0,sample.size().h - new_height)
        if keep_aspect:
            original_aspect = sample.size()
            wh = original_aspect.w / original_aspect.h
            new_width = int(wh * new_height)
            max_x = original_aspect.w - new_width
            if new_x > max_x: new_x=int(max_x)
        if not need_box:
            return sample.crop(new_x,new_y,new_width,new_height, overlap_to_keep)
        else:
            while True:
                new_width=random.randint(min_width,sample.size().w)
                new_height=random.randint(min_height,sample.size().h)
                new_x  = random.randint(0,sample.size().w - new_width)
                new_y  = random.randint(0,sample.size().h - new_height)
                new_sample = sample.crop(new_x,new_y,new_width,new_height, overlap_to_keep)
                if keep_aspect:
                    original_aspect = sample.size()
                    wh = original_aspect.w / original_aspect.h
                    new_width = int(wh * new_height)
                    max_x = original_aspect.w - new_width
                    if new_x > max_x: new_x=int(max_x)
                if len(new_sample.detection.boxes2d) > 0:
                    return new_sample

    if isinstance(sample,list):
        return [scaleValue(x) for x in sample]
    return scaleValue(sample)
class RandomCutTransform():
    def __init__(self,min_width:int,min_height:int, overlap_to_keep=0.2, need_box = True):
        self.min_width = min_width
        self.min_height = min_height
        self.overlap_to_keep = overlap_to_keep
        self.need_box = need_box
    def __call__(self,sample: Union[Sample,List[Sample]]):
        return random_cut(sample,self.min_width,self.min_height, self.overlap_to_keep, self.need_box)
class RandomCropAspectTransform():
    def __init__(self,min_width:int,min_height:int, overlap_to_keep=0.2, need_box = True):
        self.min_width = min_width
        self.min_height = min_height
        self.overlap_to_keep = overlap_to_keep
        self.need_box = need_box
    def __call__(self,sample: Union[Sample,List[Sample]]):
        return random_cut(sample,self.min_width,self.min_height, self.overlap_to_keep, self.need_box, keep_aspect = True)