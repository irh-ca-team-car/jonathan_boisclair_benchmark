from ..datasets.Sample import Size
from ..detectors.Detector import *
from ..detectors.Detection import *
from .EffDet.model import getEfficientDetImpl
from ..datasets.detection import DetectionDataset
from .YoloV7 import *
from .YoloV8 import *
import torch


class EfficientDetector(Detector):
    module: torch.nn.Module
    dataset: DetectionDataset
    isTrain: bool

    def __init__(self,compound_coef) -> None:
        super(EfficientDetector, self).__init__(3, True)
        self.compound_coef = compound_coef
        self.module = getEfficientDetImpl(compound_coef=compound_coef)
        self.dataset = DetectionDataset.named("coco-empty")

        import pathlib
        import os
        weights_path = os.path.join(pathlib.Path(__file__).parent.absolute(),f'../datasets/coco/efficientdet-d{compound_coef}.pth') 
        try:
            keys = self.module.model.load_state_dict(torch.load(weights_path), strict=False)
            print("Loaded EfficientDet weights, except",keys, weights_path)
        except:
            print("Could not load weights",weights_path, "did you download them in the coco folder")
            pass
        self.freeze_backbone()

    def parameters(self, recurse: bool = True):
        return self.module.model.parameters()

    def state_dict(self):
        return self.module.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = False):
        return self.module.model.load_state_dict(self, state_dict, strict)
    def half(self):
        self.module=self.module.half()
    def float(self):
        self.module=self.module.float()
    def train(self):
        self.isTrain = True

    def eval(self):
        self.isTrain = False
    def to(self,device:torch.device):
        super(EfficientDetector,self).to(device)
        self.module = self.module.to(device)
        return self
    def _forward(self, rgb:torch.Tensor,lidar:torch.Tensor,thermal:torch.Tensor, target=None, dataset=None):
        from .EffDet.utils.utils import postprocess,invert_affine
        from .EffDet.efficientdet.utils import BBoxTransform, ClipBoxes

        if isinstance(rgb,list):
            return self._forward(torch.cat([v.unsqueeze(0) for v in rgb],0), None,None, target)
        if len(rgb.shape)==3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.to(device=self.device)
        
        if(target is not None):
            if isinstance(target,list):
                target = [t.detection.toX1Y1X2Y2C(self.device) for t in target]
            else:
                target = [target.detection.toX1Y1X2Y2C(self.device)]
            loss_dict= self.module(rgb,target,self.dataset.classesList() )
            return sum(loss for loss in loss_dict)
        #print(torchvisionresult.__class__, torchvisionresult)
        _, regression, classification, anchors = self.module.model(rgb)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(rgb,
                          anchors.detach(), regression.detach(), classification.detach(),
                          regressBoxes, clipBoxes,
                          0.2, 0.2)
        #out = invert_affine(framed_metas, out)
        
        ret = []
        for res in out:
            det = Detection()
            for b in range(res["rois"].shape[0]):
                box = Box2d()
                box.x = res["rois"][b, 0].item()
                box.y = res["rois"][b, 1].item()
                box.w = res["rois"][b, 2].item()-res["rois"][b, 0].item()
                box.h = res["rois"][b, 3].item()-res["rois"][b, 1].item()
                box.c = res["class_ids"][b].item()
                box.cf = res["scores"][b].item()
                box.cn = str(box.c)#CocoDetection.getName(box.c)
                if dataset is not None:
                    box.cn = dataset.getName(box.c)
                else:
                    box.cn = self.dataset.getName(box.c)
                det.boxes2d.append(box)
            ret.append(det)

        if len(ret) == 1:
            return ret[0]
        if len(ret) == 0:
            return None
        return ret

        #torch.save(torchvisionresult, "test.pt")
        exit(0)
        return Detection()
        #return Detection.fromTorchVision(torchvisionresult, dataset)

        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()

        loss = cls_loss + reg_loss

    def freeze_backbone(self):
        def freeze_backbone_(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False
            classname = m.__class__.__name__
        self.module.apply(freeze_backbone_)
    def unfreeze_backbone(self):
        def unfreeze_backbone_(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = True
            classname = m.__class__.__name__
        self.module.apply(unfreeze_backbone_)

    def adaptTo(self, dataset):
        state = self.module.state_dict()
        self.module = getEfficientDetImpl(num_class=len(dataset.classesList()),compound_coef=self.compound_coef)
        try:
            self.module.load_state_dict(state,strict=False)
        except:
            pass
        self.dataset = dataset
        return self
    def calculateLoss(self,sample:Sample):
        if isinstance(sample,list):
            sample=[s.scale(Size(512,512)) for s in sample]
        else:
            sample= sample.scale(Size(512,512))

        losses= self.forward(sample, sample)
        return losses

  
class EfficientDetectorInitiator():
    def __init__(self,coef):
        self.coef=coef
        pass
    def __call__(self):
        return EfficientDetector(self.coef)
    
for c in range(9):
    Detector.register("EfficientDetector_d"+str(c), EfficientDetectorInitiator(c))


