from typing import List, Type, Union
import torch.nn as nn
import torch
from ..datasets import Sample, Size

class ITI(nn.Module):
    registered={}
    name: Union[str,None]
    def __init__(self, supportBatch) -> None:
        super(ITI,self).__init__()
        self.supportBatch = supportBatch
        self.name=None
    def forward(self,x:Union[Sample,List[Sample]]) -> Union[Sample,List[Sample]]:
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
    def named(name) -> Type["ITI"]:
        if isinstance(name,ITI):
            return ITI.named(name)
        return ITI.registered[name]
    def allRegistered():
        return dict(ITI.registered)
    def loss(self,s1:Sample, s2:Sample) -> torch.Tensor:
        device = s1.getRGB().device
        img1 = s1.getRGB()
        img2=  s2.getRGB()
        loss_fn = torch.nn.HuberLoss().to(device)
        return loss_fn(img1,img2)
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
        output[output>0.9999] = 0.9999
        output[output<0.0001] = 0.0001
        #output = torch.clamp(output,0,1)
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


class DenseFuse_ITI(ITI):
    def __init__(self) -> None:
        super(DenseFuse_ITI,self).__init__(False)
        from .impl.densefuse_pytorch.utils import test_rgb
        from .impl.densefuse_pytorch.densefuse_net import DenseFuseNet
        self.model = DenseFuseNet()
        self.fusion_phase = test_rgb()

        import pathlib
        import os
        weights_path = os.path.join(pathlib.Path(__file__).parent.absolute(),f'impl/densefuse_pytorch/train_result/model_weight.pkl') 
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu"))['weight'])
        self.device = "cpu"
        

    def to(self,device):
        self.model = self.model.to(device)
        self.fusion_phase = self.fusion_phase.to(device)
        self.device=device
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
        mode='add'
        x.to(self.device)
        window_width=1
        output= self.fusion_phase.get_fusion(
            x.getRGB(),x.getThermal(),self.model,
                   save_path = None, 
                   save_name = None, 
                   mode=mode,window_width=window_width)
        #output = self.model(img.to(self.device).unsqueeze(0))
        #output = torch.clamp(output,0,1)
        #print(output.shape)
        out = Sample()
        out.setImage(output.to(self.device).squeeze(0))
        out.detection = x.detection
        return out
    def loss(self,s1:Sample, s2:Sample):
        from .impl.densefuse_pytorch.ssim import SSIM 
        device = s1.getRGB().device
        MSE_fun = nn.MSELoss().to(device)
        SSIM_fun = SSIM().to(device)

        img1 = s1.getRGB().unsqueeze(0)
        img2=  s2.getRGB().unsqueeze(0)

        mse_loss = MSE_fun(img1,img2)
        ssim_loss = 1-SSIM_fun(img1,img2)
        loss = mse_loss+ssim_loss
        return loss
       

ITI.register("DenseFuse", DenseFuse_ITI)
