

from typing import List, Optional, Union
from ..datasets import Sample,Size


def scale(sample: Union[Sample,List[Sample]],width:Union[int,Size],height:Union[int,None]=None, stretch=True) -> Union[Sample,List[Sample]]:
    if isinstance(width,Size):
        return scale(sample, width.w,width.h, stretch)
    def scaleValue(sample:Sample):
        if stretch:
            return sample.scale(Size(width,height))
        else:
            raise Exception("Not implemented yet")
        return sample

    if isinstance(sample,list):
        return [scaleValue(x) for x in sample]
    return scaleValue(sample)
class ScaleTransform():
    def __init__(self,width:Union[int,Size],height:Union[int,None]=None, stretch=True):
        self.width = width
        self.height = height
        self.stretch = stretch
    def __call__(self,sample: Union[Sample,List[Sample]]):
        return scale(sample,self.width, self.height,self.stretch)