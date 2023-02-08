

from typing import List, Optional, Tuple, Union
from ..datasets import Sample,Size,Detection, Box2d
import random
import torchvision.transforms.functional
import math
import numpy as np

def rotate(sample: Union[Sample,List[Sample]], angle=None, choices = None) -> Union[Sample,List[Sample]]:
    def rotate_point(p, origin=(0, 0), degrees=0) -> Tuple[float,float]:
        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)
    if choices is None:
        choices = [0,90,180,270]
    def scaleValue(sample:Sample):
        rotation = angle
        if rotation is None:
                       rotation = random.choices(choices)[0]
        sample = sample.clone()
        rgb = sample.getRGB()
        
        if sample.detection is not None:
            sample_size = sample.size()
            center = sample.size().div(2)
            center = (center.w,center.h)

            X1Y1X2Y2CFC=sample.detection.toX1Y1X2Y2CFC()
            boxes = list(sample.detection.boxes2d)
            sample.detection.boxes2d.clear()
            for b in range(X1Y1X2Y2CFC.shape[0]):
                box = X1Y1X2Y2CFC[b]
                xy = (int(box[0]),int(box[1]))
                x2y2 = (int(box[2]),int(box[3]))
                xy2 = (int(box[0]),int(box[3]))
                x2y = (int(box[2]),int(box[1]))
                cf = float(box[4])
                c = int(box[5])
                cn = boxes[b].cn

                xy = rotate_point((xy),center,rotation)
                x2y2 = rotate_point((x2y2),center,rotation)
                xy2 = rotate_point((xy2),center,rotation)
                x2y = rotate_point((x2y),center,rotation)

                newBox = Box2d()
                newBox.cf = cf
                newBox.c = c
                newBox.cn = cn

                x = min(xy[0],x2y2[0],xy2[0],x2y[0])
                y = min(xy[1],x2y2[1],xy2[1],x2y[1])

                w = max(xy[0],x2y2[0],xy2[0],x2y[0]) - x
                h = max(xy[1],x2y2[1],xy2[1],x2y[1]) - y

                x_=x
                y_=y
                x = max(0,x)
                y = max(0,y)
                loss_left = x-x_
                loss_top = y-y_

                loss_right = max(0,x+w- (sample_size.w))
                loss_bottom= max(0,y+h- (sample_size.h))

                newBox.x = x
                newBox.y = y
                newBox.w = w - loss_left - loss_right
                newBox.h = h - loss_top - loss_bottom

                if newBox.surface() < 10:
                    continue

                sample.detection.boxes2d.append(newBox)
        sample.setImage(torchvision.transforms.functional.rotate(
            rgb,rotation
        ))
        return sample

    if isinstance(sample,list):
        return [scaleValue(x) for x in sample]
    return scaleValue(sample)

class RandomRotateTransform():
    def __init__(self, choices = None):
        self.choices = choices
    def __call__(self,sample: Union[Sample,List[Sample]]):
        return rotate(sample, angle=None, choices = self.choices)

class RotateTransform():
    def __init__(self, rotation):
        self.rotation = rotation
    def __call__(self,sample: Union[Sample,List[Sample]]):
        return rotate(sample, angle=self.rotation)