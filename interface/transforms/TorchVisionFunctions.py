
from typing import List, Optional, Union
from ..datasets import Sample,Size
import torchvision.transforms.functional
class AdjustBrightness():
    def __init__(self,brightness_factor):
        self.brightness_factor = brightness_factor
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_brightness(sample.getRGB(),self.brightness_factor)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AdjustContrast():
    def __init__(self,contrast_factor):
        self.contrast_factor = contrast_factor
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_contrast(sample.getRGB(),self.contrast_factor)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AdjustContrast():
    def __init__(self,gamma,gain):
        self.gamma = gamma
        self.gain = gain
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_gamma(sample.getRGB(),self.gamma,self.gain)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AdjustHue():
    def __init__(self,hue_factor):
        self.hue_factor = hue_factor
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_hue(sample.getRGB(),self.hue_factor)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AdjustSaturation():
    def __init__(self,saturation_factor):
        self.saturation_factor = saturation_factor
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_saturation(sample.getRGB(),self.saturation_factor)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AdjustSaturation():
    def __init__(self,sharpness_factor):
        self.sharpness_factor = sharpness_factor
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.adjust_sharpness(sample.getRGB(),self.sharpness_factor)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class AutoContrast():
    def __init__(self):
        pass
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.autocontrast(sample.getRGB())
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class GaussianBlur():
    def __init__(self, kernel_size: List[int], sigma: Union[List[float],None] = None):
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.gaussian_blur(sample.getRGB(),self.kernel_size,self.sigma)
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class Invert():
    def __init__(self):
        pass
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.invert(sample.getRGB())
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
class Grayscale():
    def __init__(self):
        pass
    def __call__(self,sample: Union[Sample,List[Sample]]):
        def scaleValue(sample:Sample):
            rgb = torchvision.transforms.functional.to_grayscale(sample.getRGB())
            sample.setImage(rgb)
            return sample
        if isinstance(sample,list):
            return [scaleValue(x) for x in sample]
        return scaleValue(sample)
import torch
class Cat():
    def __init__(self):
        pass
    def __call__(self,sample: Union[Sample,List[Sample]]) -> torch.Tensor:
        if isinstance(sample,list):
            return torch.cat([v.getRGB().unsqueeze(0) for v in sample],0)
        else:
            return sample.getRGB().unsqueeze(0)
class ThermalCat():
    def __init__(self):
        pass
    def __call__(self,sample: Union[Sample,List[Sample]]) -> torch.Tensor:
        if isinstance(sample,list):
            return torch.cat([v.getRGBT().unsqueeze(0) for v in sample],0)
        else:
            return sample.getRGBT().unsqueeze(0)