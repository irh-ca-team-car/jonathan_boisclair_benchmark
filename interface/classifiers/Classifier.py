from typing import Callable, Dict, List, Type, Union
import torch
import torch.nn as nn
from ..datasets.Sample import Sample,Classification
from ..datasets.classification import ClassificationDataset, ImageNetDataset
import torchvision

class Classifier(nn.Module):
    registered_classifiers: Dict[str,Callable[[],"Classifier"]] = dict()
    def __init__(self,num_channel:int, support_batch):
        super(Classifier,self).__init__()
        self.num_channel = num_channel
        self.support_batch = support_batch
        self.device = torch.device("cpu")

    def forward(self, x:Union[Sample, List[Sample]], target=None, dataset=None) -> Union[Classification,List[Classification]]:
        if dataset is None:
            from ..datasets.classification import ClassificationDataset
            dataset = ClassificationDataset.named("IMAGENET1K_V1")
        if not isinstance(x,Sample) and not isinstance(x,list):
            raise Exception("Argument is not a Sample")
        if isinstance(x,list):
            for v in x:
                if not isinstance(v,Sample):
                    raise Exception("Argument list contains non samples")
            if not self.support_batch:
                if target is not None:
                    return sum([self.forward(v[0],target=v[1], dataset=dataset) for v in zip(x,target)])
                
                return [self.forward(v, dataset=dataset) for v in x]
            else:
                if self.num_channel ==1 :
                    return self._forward([v.getGray() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
                if self.num_channel ==3 :
                    return self._forward([v.getRGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
                if self.num_channel ==4 :
                    return self._forward([v.getARGB() for v in x],[v.getLidar() for v in x],[v.getThermal() for v in x],target, dataset=dataset)
        
        if self.num_channel ==1 :
            return self._forward(x.getGray(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        if self.num_channel ==3 :
            return self._forward(x.getRGB(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        if self.num_channel ==4 :
            return self._forward(x.getARGB(), x.getLidar(), x.getThermal(),target, dataset=dataset)
        return x
    def adaptTo(self,dataset:ClassificationDataset):
        print("Adapting to ",dataset.getName())
        return self
    def eval(self):
        return self
    def train(self):
        return self
    def save(self,file) ->None:
        torch.save(self.state_dict(),file)
    def load(self,file)->None:
        state_dict= torch.load(file, map_location=self.device)
        try:
            self.load_state_dict(state_dict, strict = False)
        except:
            pass
    def to(self,device:torch.device):
        super(Classifier,self).to(device)
        self.device = device
        return self
    def register(name:str,objClass) -> None:
        Classifier.registered_classifiers[name]=objClass
    def named(name:str) -> "Classifier":
        c= Classifier.registered_classifiers[name]()
        return c
    def getAllRegisteredDetectors() -> Dict[str,Type["Classifier"]]:
        return dict(Classifier.registered_classifiers)
    def calculateLoss(self,sample:Union[Sample, List[Sample]])->torch.Tensor:
        huber = torch.nn.HuberLoss().to(self.device)

        result = self.forward(sample)
        if isinstance(sample,Sample):
            return huber.forward(result.confidences.to(self.device), sample.classification.confidences.to(self.device))
        return sum([
            huber.forward(a.confidences.to(self.device), b.classification.confidences.to(self.device))
            for a,b in zip(result,sample)
        ])

class TorchVisionClassifier(Classifier):
    model:torch.nn.Module
    def __init__(self, initiator,w, num_classes=None, **kwargs):
        super(TorchVisionClassifier,self).__init__(3,True)
        self.initiator = initiator
        self.w=w
        self.kwargs=kwargs
        if num_classes is not None:
            self.model:torch.nn.Module = initiator(num_classes=num_classes, **kwargs)
        else:
            self.model = initiator(weights=w,**kwargs)
        self.model.eval()
        self.dataset = ClassificationDataset.named("IMAGENET1K_V1")
    def adaptTo(self,dataset) -> "TorchVisionClassifier":
        if self.dataset != dataset:
            print("Torchvision model adapting to ",dataset.getName())
            newModel = TorchVisionClassifier(self.initiator,self.w, num_classes=len(dataset.classesList()), **self.kwargs) 
            try:
                newModel.load_state_dict(self.state_dict(),strict=False)
            except:
                pass
            print("New dataset has ",len(dataset.classesList()),"classes")
            newModel.kwargs = self.kwargs
            newModel.initiator = self.initiator
            newModel.w = self.w
            newModel.dataset = dataset
            return newModel
        else:
            return self
    def _forward(self, rgb:Union[torch.Tensor,List[torch.Tensor]],lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None) -> Union[Classification,List[Classification]]:
        if isinstance(rgb, List):
            input = rgb
            rgb = torch.cat([t.unsqueeze(0) for t in rgb],0).to(self.device)
            output = self.model.to(self.device).forward(rgb)
            return [Classification(output[i],self.dataset) for i in range(len(input))]

        output = self.model(rgb.unsqueeze(0)).to(self.device)
        c = Classification(output[0],self.dataset)
        return c


    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    def to(self,device:torch.device):
        super(TorchVisionClassifier,self).to(device)
        self.model = self.model.to(device)
        return self
   
class TorchVisionClassifierInitiator():
    def __init__(self,initiator,w, **kwargs):
        self.initiator = initiator
        self.w=w
        self.kwargs = kwargs
        pass
    def __call__(self):
        return TorchVisionClassifier(self.initiator,self.w,None, **self.kwargs)

TorchVisionClassifier.register("alexnet", TorchVisionClassifierInitiator(torchvision.models.alexnet, torchvision.models.AlexNet_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("densenet121", TorchVisionClassifierInitiator(torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("densenet161", TorchVisionClassifierInitiator(torchvision.models.densenet161, torchvision.models.DenseNet161_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("densenet169", TorchVisionClassifierInitiator(torchvision.models.densenet169, torchvision.models.DenseNet169_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("densenet201", TorchVisionClassifierInitiator(torchvision.models.densenet201, torchvision.models.DenseNet201_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("convnext_base", TorchVisionClassifierInitiator(torchvision.models.convnext_base, torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("convnext_large", TorchVisionClassifierInitiator(torchvision.models.convnext_large, torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("convnext_small", TorchVisionClassifierInitiator(torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("convnext_tiny", TorchVisionClassifierInitiator(torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("efficientnet_b0", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b1", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b2", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b3", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b4", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b5", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b5, torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b6", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b6, torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("efficientnet_b7", TorchVisionClassifierInitiator(torchvision.models.efficientnet_b7, torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("googlenet", TorchVisionClassifierInitiator(torchvision.models.googlenet, torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1, init_weights=False))
TorchVisionClassifier.register("inception_v3", TorchVisionClassifierInitiator(torchvision.models.inception_v3, torchvision.models.Inception_V3_Weights.IMAGENET1K_V1, init_weights=False))
TorchVisionClassifier.register("vit_l_32", TorchVisionClassifierInitiator(torchvision.models.vit_l_32, torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vit_b_16", TorchVisionClassifierInitiator(torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vit_b_32", TorchVisionClassifierInitiator(torchvision.models.vit_b_32, torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vit_l_16", TorchVisionClassifierInitiator(torchvision.models.vit_l_16, torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("vgg11", TorchVisionClassifierInitiator(torchvision.models.vgg11, torchvision.models.VGG11_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vgg13", TorchVisionClassifierInitiator(torchvision.models.vgg13, torchvision.models.VGG13_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vgg16", TorchVisionClassifierInitiator(torchvision.models.vgg16, torchvision.models.VGG16_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("vgg19", TorchVisionClassifierInitiator(torchvision.models.vgg19, torchvision.models.VGG19_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("maxvit_t", TorchVisionClassifierInitiator(torchvision.models.maxvit_t, torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("shufflenet_v2_x0_5", TorchVisionClassifierInitiator(torchvision.models.shufflenet_v2_x0_5, torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("shufflenet_v2_x1_0", TorchVisionClassifierInitiator(torchvision.models.shufflenet_v2_x1_0, torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("shufflenet_v2_x1_5", TorchVisionClassifierInitiator(torchvision.models.shufflenet_v2_x1_5, torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("shufflenet_v2_x2_0", TorchVisionClassifierInitiator(torchvision.models.shufflenet_v2_x2_0, torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("mobilenet_v3_large", TorchVisionClassifierInitiator(torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("mobilenet_v3_small", TorchVisionClassifierInitiator(torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1))

TorchVisionClassifier.register("resnet18", TorchVisionClassifierInitiator(torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("resnet34", TorchVisionClassifierInitiator(torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("resnet50", TorchVisionClassifierInitiator(torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("resnet101", TorchVisionClassifierInitiator(torchvision.models.resnet101, torchvision.models.ResNet101_Weights.IMAGENET1K_V1))
TorchVisionClassifier.register("resnet152", TorchVisionClassifierInitiator(torchvision.models.resnet152, torchvision.models.ResNet152_Weights.IMAGENET1K_V1))



