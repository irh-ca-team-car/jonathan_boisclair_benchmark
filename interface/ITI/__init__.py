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
    @staticmethod
    def named(name) -> Type["ITI"]:
        try:
            if name in ITI.registered:
                return ITI.registered[name]
            else:
                return ITI_Identity
        except:
            return None
    @staticmethod
    def allRegistered():
        return dict(ITI.registered)
    @staticmethod
    def loss(s1:Sample, s2:Sample) -> torch.Tensor:
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
    def to(self,device,*c,**k):
        
        self.dummyParameter.to(device,*c,**k)
        return self
class CAE_ITI(ITI):
    def __init__(self, model) -> None:
        super(CAE_ITI,self).__init__(True)
        self.model = model
        self.device = "cpu"
    def to(self,device, *c,**k):
        self.device = device
        self.model = self.model.to(device, *c,**k)
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
        

    def to(self,device, *c,**k):
        self.model = self.model.to(device, *c,**k)
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



import segmentation_models_pytorch as smp

class SegModITI(ITI):
    model:torch.nn.Module
    def __init__(self, initiator,weights=None, num_output=1, **kwarg):
        super(SegModITI,self).__init__(True)
        self.weights = weights
        self.initiator = initiator
        self.kwarg = kwarg
        self.model:torch.nn.Module = initiator(num_output=num_output, weights=weights, **self.kwarg)
        self.model.eval()

    def _forward(self, input:Union[Sample,List[Sample]])->Union[Sample,List[Sample]]:
        if isinstance(input,list):
            is_list = True
        else:
            input = [input]
            is_list = False
        
        if self.kwarg["in_channels"] ==4:
            rgb = torch.cat([x.getRGBT().unsqueeze(0) for x in input],0)
        elif self.kwarg["in_channels"] ==1:
            rgb = torch.cat([x.getGray().unsqueeze(0) for x in input],0)
        else: 
            rgb = torch.cat([x.getRGB().unsqueeze(0) for x in input],0)

        rgb = rgb.to(device=self.device)
        
        torchvisionresult= self.model.to(self.device).forward(rgb.to(self.device))

        result=[]
        for i in range(torchvisionresult.shape[0]):# self.weights.meta["categories"]
            s = Sample()
            s.setImage(torchvisionresult[i])
            result.append(s)
        if not is_list:
            return result[0]
        
        return result
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(ITI,self).to(device)
        self.model = self.model.to(device)
        self.device = device
        return self
    def loss(self,sample:Sample, sample2:Sample):
        MSE_fun = nn.MSELoss().to(self.device)
        return MSE_fun(sample._img, sample2._img)
    
    def freeze_backbone(self):
        for name,p in (self.named_parameters()):
            if "encoder" in name:
                p.requires_grad_(False)
        
    def unfreeze_backbone(self):
        for name,p in self.named_parameters():
            p.requires_grad_(True)
    @staticmethod
    def optimizer(model, lr=2e-3, lr_encoder=2e-6):
        g = [], []
        for v in model.modules():
            for p_name, p in v.named_parameters(recurse=0):
                if 'encoder' in p_name:  
                    g[1].append(p)
                else:
                    g[0].append(p) 
        optim= torch.optim.Adam(g[0],lr=lr)
        optim.add_param_group({'params': g[1], 'lr': lr_encoder})  # add g1 (BatchNorm2d weights)
        
        return optim
    
class SegModITIInitiator():
    def __init__(self,model,encoder_name, encoder_weights, **kwarg):
        self.kwarg = kwarg
        self.model = model
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        pass
    def initiator(self,test_run=False, **kwargs):
        num_output = 1
        in_channels = 3
        if "num_output" in kwargs:
            num_output = kwargs["num_output"]
        if "in_channels" in kwargs:
            in_channels = kwargs["in_channels"]
        if self.model == "unet":
            return smp.Unet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "unet++":
            return smp.UnetPlusPlus(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "manet":
            return smp.MAnet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "linknet":
            return smp.Linknet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "pspnet":
            return smp.PSPNet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "pan":
            return smp.PAN(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "deeplabv3":
            return smp.DeepLabV3(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "deeplabv3+":
            return smp.DeepLabV3Plus(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        if self.model == "fpn":
            return smp.FPN(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                           in_channels=in_channels,
                            classes= num_output
                            )
        
        return None
    def __call__(self):
        return SegModITI(self.initiator,**self.kwarg)
    def test(self):
        try:
            #self.initiator(True)
            return True
        except:
            return False

archs = [
    "unet","unet++","manet","linknet","pspnet","pan","deeplabv3","deeplabv3+","fpn"
]
encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "timm-resnest14d",
    "timm-resnest26d",
    "timm-resnest50d",
    "timm-resnest101e",
    "timm-resnest200e",
    "timm-resnest269e",
    "timm-resnest50d_4s2x40d",
    "timm-resnest50d_1s4x24d",
    "timm-res2net50_26w_4s",
    "timm-res2net101_26w_4s",
    "timm-res2net50_26w_6s",
    "timm-res2net50_26w_8s",
    "timm-res2net50_48w_2s",
    "timm-res2net50_14w_8s",
    "timm-res2next50",
    "timm-regnetx_002",
    "timm-regnetx_004",
    "timm-regnetx_006",
    "timm-regnetx_008",
    "timm-regnetx_016",
    "timm-regnetx_032",
    "timm-regnetx_040",
    "timm-regnetx_064",
    "timm-regnetx_080",
    "timm-regnetx_120",
    "timm-regnetx_160",
    "timm-regnetx_320",
    "timm-regnety_002",
    "timm-regnety_004",
    "timm-regnety_006",
    "timm-regnety_008",
    "timm-regnety_016",
    "timm-regnety_032",
    "timm-regnety_040",
    "timm-regnety_064",
    "timm-regnety_080",
    "timm-regnety_120",
    "timm-regnety_160",
    "timm-regnety_320",
    "timm-gernet_s",
    "timm-gernet_m",
    "timm-gernet_l",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "timm-skresnet18",
    "timm-skresnet34",
    "timm-skresnext50_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "inceptionresnetv2",
    "inceptionv4",
    "xception",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "timm-efficientnet-b0",
    "timm-efficientnet-b1",
    "timm-efficientnet-b2",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b5",
    "timm-efficientnet-b6",
    "timm-efficientnet-b7",
    "timm-efficientnet-l2",
    "timm-efficientnet-lite0",
    "timm-efficientnet-lite1",
    "timm-efficientnet-lite2",
    "timm-efficientnet-lite3",
    "timm-efficientnet-lite4",
    "mobilenet_v2",
    "timm-mobilenetv3_large_075",
    "timm-mobilenetv3_large_100",
    "timm-mobilenetv3_large_minimal_100",
    "timm-mobilenetv3_small_075",
    "timm-mobilenetv3_small_100",
    "timm-mobilenetv3_small_minimal_100",
    "dpn68",
    "dpn98",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
    "mit_b5",
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
    "mobileone_s3",
    "mobileone_s4s",
    "tu-maxvit_base_tf_224",
    "tu-maxvit_base_tf_384",
    "tu-maxvit_base_tf_512",
    "tu-maxvit_large_tf_224",
    "tu-maxvit_large_tf_384",
    "tu-maxvit_large_tf_512",
    "tu-maxvit_nano_rw_256",
    "tu-maxvit_pico_rw_256",
    "tu-tinynet_a",
    "tu-tinynet_b",
    "tu-tinynet_c",
    "tu-tinynet_d",
    "tu-tinynet_e",
]
for arch in archs:
    for encoder in encoders:
        ITI.register(arch+"+"+encoder+"_1->1",SegModITIInitiator(arch, encoder, "imagenet",in_channels=1, num_output = 1))
        ITI.register(arch+"+"+encoder+"_3->1",SegModITIInitiator(arch, encoder, "imagenet",in_channels=3,num_output = 1))
        ITI.register(arch+"+"+encoder+"_4->1",SegModITIInitiator(arch, encoder, "imagenet",in_channels=4,num_output = 1))

        ITI.register(arch+"+"+encoder+"_1->3",SegModITIInitiator(arch, encoder, "imagenet",in_channels=1, num_output = 3))
        ITI.register(arch+"+"+encoder+"_3->3",SegModITIInitiator(arch, encoder, "imagenet",in_channels=3,num_output = 3))
        ITI.register(arch+"+"+encoder+"_4->3",SegModITIInitiator(arch, encoder, "imagenet",in_channels=4,num_output = 3))

