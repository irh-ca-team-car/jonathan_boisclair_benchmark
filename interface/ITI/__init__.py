from typing import List
import torch.nn as nn
import torch
from ..datasets import Sample, Size

class ITI(nn.Module):
    registered={}
    def __init__(self, supportBatch) -> None:
        super(ITI,self).__init__()
        self.supportBatch = supportBatch
    def forward(self,x:Sample) -> Sample:
        if isinstance(x,list):
            if self.supportBatch:
                return self._forward(x)
            else:
                return [self._forward(v) for v in x]
        elif isinstance(x,Sample):
            return self._forward(x)
        else:
            return x
    def _forward(self,x:Sample) -> Sample:
        return x
    def register(name,clz):
        if isinstance(name,ITI):
            return ITI.register(name,clz)
        ITI.registered[name]=clz
    def named(name):
        if isinstance(name,ITI):
            return ITI.named(name)
        return ITI.registered[name]
    def allRegistered():
        return dict(ITI.registered)
from .impl.CAEbase import VCAE5,VCAE6
class ITI_Identity(ITI):
    def __init__(self) -> None:
        super(ITI_Identity,self).__init__(True)
        self.dummyParameter = torch.nn.Parameter(torch.tensor(1.0))
    def _forward(self, x: Sample) -> Sample:
        return x
class CAE_ITI(ITI):
    def __init__(self, model) -> None:
        super(CAE_ITI,self).__init__(True)
        self.model = model
        self.device = "cpu"
    def to(self,device):
        self.device = device
        self.model = self.model.to(device)
        return self
    def getRGBT(x: Sample) -> torch.Tensor:
        img=None
        thermal = None
        if x.hasImage():
            img = x.getRGB()
        if x.hasThermal():
            thermal = x.getThermal()
        if thermal is not None:
            if img.shape[1:] != thermal.shape[1:]:
                x=x.clone().scale(Size(img.shape[1],img.shape[2]))
                thermal = x.getThermal()
        else:
            thermal = torch.zeros(1,*img.shape[1:]).to(img.device)
        tensor = torch.cat([img,thermal],0)
        img = tensor
        return img
    def _forward(self, x: Sample) -> Sample:
        if isinstance(x,list):
            #handle batch
            size = x[0].size()
            samps : List[Sample] = [v.clone().scale(size) for v in x]

            batched = torch.cat([
                CAE_ITI.getRGBT(v).unsqueeze(0) for v in samps
            ],0)
            output = self.model(batched.to(self.device))
            output = torch.clamp(output,0,1)
            ret=[]
            for i in range(output.shape[0]):
                tmp = Sample()
                tmp.setImage(output[i])
                tmp.detection = x[i].detection
                ret.append(tmp)
            return ret
        img = CAE_ITI.getRGBT(x)
        output = self.model(img.to(self.device).unsqueeze(0))
        output = torch.clamp(output,0,1)
        #print(output.shape)
        out = Sample()
        out.setImage(output.squeeze(0))
        out.detection = x.detection
        return out

class CAE_ITIInitiator():
    def __init__(self,initiator):
        self.initiator=initiator
    def __call__(self):
        return CAE_ITI(self.initiator("cpu"))

ITI.register("VCAE6", CAE_ITIInitiator(VCAE6))
ITI.register("Identity", ITI_Identity)