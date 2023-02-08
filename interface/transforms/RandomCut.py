

from typing import List, Optional, Union
from ..datasets import Sample,Size
import random

def random_cut(sample: Union[Sample,List[Sample]],min_width:int,min_height:int, overlap_to_keep=0.2) -> Union[Sample,List[Sample]]:
    def scaleValue(sample:Sample):
        new_width=random.randint(min_width,sample.size().w)
        new_height=random.randint(min_height,sample.size().h)
        new_x  = random.randint(0,sample.size().w - new_width)
        new_y  = random.randint(0,sample.size().h - new_height)

        return sample.crop(new_x,new_y,new_width,new_height, overlap_to_keep)

    if isinstance(sample,list):
        return [scaleValue(x) for x in sample]
    return scaleValue(sample)