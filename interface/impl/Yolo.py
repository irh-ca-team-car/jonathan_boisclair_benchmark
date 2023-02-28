from interface.datasets.Sample import Size
from ..detectors.Detector import *
from ..detectors.Detection import *
from ..datasets import CocoDetection
import torch


class YoloV7Detector(Detector):
    module: torch.nn.Module
    dataset: CocoDetection
    isTrain: bool

    def __init__(self,compound_coef) -> None:
        super(YoloV7Detector, self).__init__(3, True)
        

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
        super(YoloV7Detector,self).to(device)
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

  
    def adaptTo(self, dataset):
        raise Exception("Not implemented")
    def calculateLoss(self,sample:Sample):
        if isinstance(sample,list):
            sample=[s.scale(Size(512,512)) for s in sample]
        else:
            sample= sample.scale(Size(512,512))

        losses= self.forward(sample, sample)
        return losses

  
class YoloV7DetectorInitiator():
    def __init__(self,coef):
        self.coef=coef
        pass
    def __call__(self):
        return YoloV7Detector(self.coef)
    