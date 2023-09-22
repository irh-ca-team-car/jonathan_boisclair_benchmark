from typing import Callable, Dict, List, Union
import torch
import torch.nn as nn
from ..datasets.Sample import Sample,Segmentation,Size
from ..datasets import DetectionDataset

class Segmenter(nn.Module):
    registered_Segmenters: Dict[str,Callable[[],"Segmenter"]] = dict()
    dataset: DetectionDataset
    def __init__(self,num_channel:int, support_batch, dataset=None):
        super(Segmenter,self).__init__()
        
        self.num_channel = num_channel
        self.support_batch = support_batch
        self.device = torch.device("cpu")
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = DetectionDataset.named("voc-2007")

    def forward(self, x:Sample, target=None) -> Segmentation:
        if self.dataset is None:
            from ..datasets import DetectionDataset
            self.dataset = DetectionDataset.named("voc-2007")
        if not isinstance(x,Sample) and not isinstance(x,list):
            raise Exception("Argument is not a Sample")
        if isinstance(x,list):
            for v in x:
                if not isinstance(v,Sample):
                    raise Exception("Argument list contains non samples"+str(v.__class__))
            if not self.support_batch:
                if target is not None:
                    return sum([self.forward(v[0],target=v[1]) for v in zip(x,target)])
                
                return [self.forward(v) for v in x]
            else:
                if self.num_channel ==1 :
                    return self._forward([v.getGray() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target)
                if self.num_channel ==3 :
                    return self._forward([v.getRGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target)
                if self.num_channel ==4 :
                    return self._forward([v.getARGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target)
        
        if self.num_channel ==1 :
            return self._forward(x.getGray(), x.getLidar(), x.getThermal(),target)
        if self.num_channel ==3 :
            return self._forward(x.getRGB(), x.getLidar(), x.getThermal(),target)
        if self.num_channel ==4 :
            return self._forward(x.getARGB(), x.getLidar(), x.getThermal(),target)
        return x
    def adaptTo(self,dataset) -> "Segmenter":
        print("Adapting to ",dataset.getName())
        self.dataset = dataset
        return self
    def eval(self) -> "Segmenter":
        return self
    def train(self) -> "Segmenter":
        return self
    def save(self,file) ->None:
        torch.save(self.state_dict(),file)
    def load(self,file)->None:
        state_dict= torch.load(file, map_location=self.device)
        try:
            self.load_state_dict(state_dict, strict = False)
        except:
            pass
    def to(self,device:torch.device) -> "Segmenter":
        super(Segmenter,self).to(device)
        self.device = device
        return self
    def registerIf(name:str,objClass) -> None:
        if not  hasattr(objClass,"test"):
            Segmenter.registered_Segmenters[name]=objClass
        else:
            if objClass.test():
                Segmenter.registered_Segmenters[name]=objClass

    def named(name:str) -> "Segmenter":
        c= Segmenter.registered_Segmenters[name]()
        return c
    def getAllRegisteredSegmenters() -> Dict[str,Callable[[],"Segmenter"]]:
        return dict(Segmenter.registered_Segmenters)
    def calculateLoss(self,sample:Union[Sample,List[Sample]])->torch.Tensor:
        ret=self.forward(sample, sample)
        if isinstance(ret,list):
            ret= sum(ret)
        return ret
    @staticmethod
    def optimizer(model):
        return torch.optim.Adamax(model.parameters())

class TorchVisionSegmenter(Segmenter):
    model:torch.nn.Module
    def __init__(self, initiator,weights=None, num_classes=None, **kwarg):
        super(TorchVisionSegmenter,self).__init__(3,True)
        self.weights = weights
        self.initiator = initiator
        self.kwarg = kwarg
        self._do_shapes = True
        if num_classes is not None:
            if weights is not None:
                self.model:torch.nn.Module = initiator(num_classes=num_classes, weights=weights, **self.kwarg)
            else:
                self.model:torch.nn.Module = initiator(num_classes=num_classes, **self.kwarg)
        else:
            self.model = initiator(weights=weights,**self.kwarg)
        self.model.eval()
    @property
    def doShape(self):
        return self._do_shapes
    @doShape.setter
    def doShape(self,value):
        self._do_shapes = value
    def adaptTo(self,dataset):
        if self.dataset != dataset:
            newModel = TorchVisionSegmenter(self.initiator,None,num_classes=len(dataset.classesList()), **self.kwarg )
            newModel.weights = self.weights
            try:
                newModel.load_state_dict(self.state_dict(),strict=False)
            except:
                pass
            newModel.dataset = dataset
            return newModel
        else:
            return self
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None)->Union[Segmentation,List[Segmentation], torch.tensor]:
        trs = self.weights.transforms(antialias=True)
        if isinstance(rgb,list):
            return self._forward(torch.cat([trs(v).unsqueeze(0) for v in rgb],0), None,None, target)
        is_list = True
        
        if len(rgb.shape)==3:
            rgb = trs(rgb).unsqueeze(0)
            is_list = False
            
        rgb = rgb.to(device=self.device)
        
        if(target is not None):
            num_class = len(self.dataset.classesList())
            if isinstance(target,list):
                target = [t.toTorchVisionSegmentationTarget(num_class, Size.fromTensor(rgb)).to(self.device) for t in target]
            else:
                target = [target.toTorchVisionSegmentationTarget(num_class,Size.fromTensor(rgb)).to(self.device)]

            torchvisionresult= self.model.to(self.device).forward(rgb.to(self.device))["out"]
            target = torch.cat([t.unsqueeze(0) for t in target],-1)
            loss = nn.CrossEntropyLoss()
            return loss.forward(torchvisionresult.permute(0,2,3,1).view(-1,num_class),target.permute(0,2,3,1).view(-1,num_class))
            #return sum(loss for loss in loss_dict.values())
        torchvisionresult= self.model.to(self.device).forward(rgb.to(self.device))
        normalized_masks = torchvisionresult["out"].softmax(dim=1)
        normalized_masks_2 = torch.max(normalized_masks, dim=1).indices.unsqueeze(1)


        result=[]
        for i in range(normalized_masks_2.shape[0]):# self.weights.meta["categories"]
            result.append(Segmentation.FromImage(normalized_masks_2[i], self.dataset.classesList()if self.doShape else None, normalized_masks[i]))
        if not is_list:
            return result[0]
        
        return result
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(Segmenter,self).to(device)
        self.model = self.model.to(device)
        self.device = device
        return self
    def calculateLoss(self,sample:Sample):
        losses= self.forward(sample, sample)
        return losses
   
class TorchVisionSegmenterInitiator():
    def __init__(self,initiator,weights, **kwarg):
        self.initiator = initiator
        self.kwarg = kwarg
        self.weights = weights
        pass
    def __call__(self):
        return TorchVisionSegmenter(self.initiator,self.weights,**self.kwarg)
    
from functools import partial
import torchvision

try:
    Segmenter.registerIf("fcn_resnet101",TorchVisionSegmenterInitiator(torchvision.models.segmentation.fcn_resnet101, weights=torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1))
    Segmenter.registerIf("fcn_resnet50",TorchVisionSegmenterInitiator(torchvision.models.segmentation.fcn_resnet50, weights=torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1))
    Segmenter.registerIf("deeplabv3_mobilenet_v3_large",TorchVisionSegmenterInitiator(torchvision.models.segmentation.deeplabv3_mobilenet_v3_large, weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1))
    Segmenter.registerIf("deeplabv3_resnet101",TorchVisionSegmenterInitiator(torchvision.models.segmentation.deeplabv3_resnet101, weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1))
    Segmenter.registerIf("deeplabv3_resnet50",TorchVisionSegmenterInitiator(torchvision.models.segmentation.deeplabv3_resnet50, weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1))
    Segmenter.registerIf("lraspp_mobilenet_v3_large",TorchVisionSegmenterInitiator(torchvision.models.segmentation.lraspp_mobilenet_v3_large, weights=torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1))
except BaseException as e:
    def format_exception(e):
        import sys
        import traceback

        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

        exception_str = "Traceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        # Removing the last \n
        exception_str = exception_str[:-1]

        return exception_str
    print("Seems like weights are not implemented, Segmentation doesn't support old pytorch versions?",format_exception(e))
  
#Incompatible for some reasons
#Detector.register("fasterrcnn_resnet50_fpn_v2",TorchVisionInitiator(torchvision.models.detection.fasterrcnn_resnet50_fpn_v2, torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1))
#Detector.register("keypointrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.keypointrcnn_resnet50_fpn, torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1))
#Detector.register("maskrcnn_resnet50_fpn",TorchVisionInitiator(torchvision.models.detection.maskrcnn_resnet50_fpn, torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1))


import segmentation_models_pytorch as smp

class SegModSegmenter(Segmenter):
    model:torch.nn.Module
    def __init__(self, initiator,weights=None, num_classes=None,dataset:DetectionDataset=None, **kwarg):
        super(SegModSegmenter,self).__init__(3,True,dataset)
        self.weights = weights
        self.initiator = initiator
        self.kwarg = kwarg
        if num_classes is not None:
            if weights is not None:
                self.model:torch.nn.Module = initiator(num_classes=num_classes, weights=weights, **self.kwarg)
            else:
                self.model:torch.nn.Module = initiator(num_classes=num_classes, **self.kwarg)
        else:
            self.model = initiator(weights=weights,num_classes=21,**self.kwarg)
        self.model.eval()

    def adaptTo(self,dataset):
        newModel = SegModSegmenter(self.initiator,None,num_classes=len(dataset.classesList()),dataset=dataset, **self.kwarg )
        newModel.weights = self.weights
        try:
            newModel.load_state_dict(self.state_dict(),strict=False)
        except:
            pass
        newModel.dataset = dataset
        return newModel
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None)->Union[Segmentation,List[Segmentation], torch.tensor]:
        if isinstance(rgb,list):
            return self._forward(torch.cat([v.unsqueeze(0) for v in rgb],0), None,None, target)
        is_list = True
        
        if len(rgb.shape)==3:
            rgb = rgb.unsqueeze(0)
            is_list = False
            
        rgb = rgb.to(device=self.device)
        
        if(target is not None):
            num_class = len(self.dataset.classesList())
            if isinstance(target,list):
                target = [t.toTorchVisionSegmentationTarget(num_class, Size.fromTensor(rgb)).to(self.device) for t in target]
            else:
                target = [target.toTorchVisionSegmentationTarget(num_class,Size.fromTensor(rgb)).to(self.device)]

            torchvisionresult= self.model.to(self.device).forward(rgb.to(self.device))
            target = torch.cat([t.unsqueeze(0) for t in target],0)
            loss1 = nn.CrossEntropyLoss()
            loss2 = nn.MSELoss()
            def loss(a,b):
                return (loss1.forward(a,b) )# + loss2.forward(a,b))/2
            return loss(torchvisionresult.permute(0,2,3,1).reshape(-1,num_class),target.permute(0,2,3,1).reshape(-1,num_class))
            #return sum(loss for loss in loss_dict.values())

        torchvisionresult= self.model.to(self.device).forward(rgb.to(self.device))
        normalized_masks = torchvisionresult.softmax(dim=1)
        normalized_masks_2 = torch.max(normalized_masks, dim=1).indices.unsqueeze(1)

        result=[]
        for i in range(normalized_masks_2.shape[0]):# self.weights.meta["categories"]
            result.append(Segmentation.FromImage(normalized_masks_2[i], self.dataset.classesList() if self.doShape else None, normalized_masks[i]))
        if not is_list:
            return result[0]
        
        return result
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(Segmenter,self).to(device)
        self.model = self.model.to(device)
        self.device = device
        return self
    def calculateLoss(self,sample:Sample):
        losses= self.forward(sample, sample)
        return losses
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
    
class SegModSegmenterInitiator():
    def __init__(self,model,encoder_name, encoder_weights, classes, **kwarg):
        self.kwarg = kwarg
        self.model = model
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.classes= classes
        pass
    def initiator(self,test_run=False, **kwargs):
        num_classes = self.classes
        if "num_classes" in kwargs:
            num_classes = kwargs["num_classes"]
        if self.model == "unet":
            return smp.Unet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "unet++":
            return smp.UnetPlusPlus(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "manet":
            return smp.MAnet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "linknet":
            return smp.Linknet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "pspnet":
            return smp.PSPNet(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "pan":
            return smp.PAN(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "deeplabv3":
            return smp.DeepLabV3(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "deeplabv3+":
            return smp.DeepLabV3Plus(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes
                            )
        if self.model == "fpn":
            return smp.FPN(self.encoder_name,
                            encoder_weights=self.encoder_weights if not test_run else None, 
                            in_channels=3,
                            classes= num_classes, encoder_depth=5
                            )
        
        return None
    def __call__(self):
        return SegModSegmenter(self.initiator,**self.kwarg)
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
        Segmenter.registerIf(arch+"+"+encoder,SegModSegmenterInitiator(arch,encoder,"imagenet",27))
