from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple, Union

from interface.datasets.Sample import Segmentation

from ..datasets.detection import DetectionDataset
from ..detectors.Detection import Detection
from ..detectors import Detector
import torch
from torchvision.ops import box_area, box_convert, box_iou


class MultiImageAveragePrecision:
    filter: Callable[[int],bool]
    def __init__(self, gt: List[Detection], val: List[Detection], verbose = False):
        self.gt = gt
        self.val = val
        self.filter = None
        self.verbose = verbose
        if len(gt) != len(val):
            raise Exception("Lenght must match")
    def __call__(self, iou:float) -> float:
        return self.calc(iou)
    def calc(self, iou: float) -> Dict[int,float]:
        keys: Set[int] = set()
        rng = range(len(self.gt))
        if self.verbose:
            from tqdm import tqdm
            rng = tqdm(rng, desc="mAP calculation", leave=False)
        if self.filter is not None:
            for i in rng: 
                gt = self.gt[i]
                det = self.val[i]
                gt.boxes2d = [box for box in gt.boxes2d if self.filter(box.c)]
                det.boxes2d = [box for box in gt.boxes2d if self.filter(box.c)]

        for i in rng:
            gt = self.gt[i]
            det = self.val[i]
            for d in gt.boxes2d:
                keys.add(d.c)
            for d in det.boxes2d:
                keys.add(d.c)
       
        ret = dict()

        for key in keys:
            pairs = []
            for i in rng:
                gt = self.gt[i]
                det = self.val[i]
                gt = gt.c(key)
                det = det.c(key)
                m = AveragePrecision(gt,det)
                pairs.append([m.precision(iou),m.recall(iou)])
            t= torch.tensor(pairs).transpose(1,0)
            precision = t[0]
            recall = t[1]
            order = recall.numpy().argsort()
            previous_recall=0
            v=0
            for x in range(len(order)):
                dr = recall[x].item()-previous_recall
                previous_recall =recall[x]
                v += dr * precision[x]
            
            ret[key] = v
        return ret
    def coco(self):
        rng = [x/100 for x in range(101)]
        data = [list(self.calc(x).values()) for x in rng]
        data = torch.tensor(data)
        data = torch.sum(data,0)
        classes = list(self.calc(0).keys())
        d = dict()

        for x in range(len(classes)):
            d[classes[x]] = data[x].item()/len(rng)

        return d
    def mAP(self,iou=None):
        if iou is not None:
            data = self.calc(iou)
        else:
            data = self.coco()
        if len(data)>0:
            return sum(data.values())/len(data)
        return 0.0

class DatasetAveragePrecision:
    filter: Callable[[int],bool]
    gt: Union[List[Detection],None]
    det: Union[List[Detection],None]
    def __init__(self, model:Detector, dataset:DetectionDataset, verbose=False):
        self.model = model
        self.dataset = dataset
        self.filter = None
        self.gt=None
        self.det = None
        self.verbose = verbose

    def __call__(self, iou:float) -> float:
        return self.calc(iou)
    def calc(self, iou: float) -> Dict[int,float]:
        if self.gt is None:
            from tqdm import tqdm
            self.gt=[]
            self.det=[]
            t = self.dataset
            if self.verbose:
                t = tqdm(t)
            for sample in t:
                self.gt.append(sample.detection)
                self.det.append(self.model(sample))
            self.map = MultiImageAveragePrecision(self.gt,self.det,self.verbose)
            self.map.filter = self.filter
        return self.map.calc(iou)

    def coco(self):
        from tqdm import tqdm

        rng = [x/100 for x in range(101)]
        if self.verbose:
            rng = tqdm(rng, desc= "coco", leave=False)
        data = [list(self.calc(x).values()) for x in rng]
        data = torch.tensor(data)
        data = torch.sum(data,0)
        classes = list(self.calc(0).keys())
        d = dict()
        for x in range(len(classes)):
            d[classes[x]] = data[x].item()/len(rng)
        return d
    def mAP(self,iou=None):
        if iou is not None:
            data = self.calc(iou)
        else:
            data = self.coco()
        if len(data)>0:
            return sum(data.values())/len(data)
        return 0.0       
        
class AveragePrecision:
    def __init__(self, gt: Detection, val: Detection):
        self.gt = gt
        self.val = val

    def calc(self, iou: float) -> float:
        classes: Dict[int, Detection] = dict()
        classesdet: Dict[int, Detection] = dict()

        for x in self.gt.boxes2d:
            if not x.c in classes:
                classes[x.c] = Detection()
            classes[x.c].boxes2d.append(x)
        for x in self.val.boxes2d:
            if not x.c in classesdet:
                classesdet[x.c] = Detection()
            classesdet[x.c].boxes2d.append(x)

        keys: Set[int] = set()
        keys = keys.union(set(classes.keys()))
        keys = keys.union(set(classesdet.keys()))
        for key in keys:
            if not key in classes:
                classes[key] = Detection()
            if not key in classesdet:
                classesdet[key] = Detection()
        if len(keys) == 0:
            return 1
        if len(keys) == 1:
            key = list(keys)[0]
            a = classes[key]
            b = classesdet[key]
            m = AveragePrecision(a, b)
            return m.precision(iou) * m.recall(iou)
        else:
            vals = []
            for key in keys:
                a = classes[key]
                b = classesdet[key]
                m = AveragePrecision(a, b)
                vals.append(m.precision(iou) * m.recall(iou))
            pass
            #vals = [AveragePrecision(a,b).precision(iou) for a,b in zip(list(classes.values()),list(classesdet.values()))]
            return sum(vals)/len(vals)
        return 0

    def precision(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        if(tp+fp > 0):
            return tp / (tp+fp)
        else:
            return 0

    def recall(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        if(tp+fn > 0):
            return tp/(tp+fn)
        else:
            return 0

    def tp(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        return tp

    def tn(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        return tn

    def fp(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        return fp

    def fn(self, iou: float) -> float:
        tp, tn, fp, fn = self.tfpn(iou)
        return fn

    def tfpn(self, iou: float) -> Tuple[float, float, float, float]:
        tp, tn, fp, fn = 0, 0, 0, 0

        gt = self.gt.toTorchVisionTarget("cpu")
        val = self.val.toTorchVisionTarget("cpu")

        gtb = [x for x in self.gt.boxes2d]
        detb = [x for x in self.val.boxes2d]

        ious = box_iou(gt["boxes"], val["boxes"])

        used_det = []

        for i, box in enumerate(gtb):
            ot = ious[i, :].numpy()
            idx = (-ot).argsort()
            found = False
            for b in range(len(ot)):
                if(ot[idx][b] >= iou and (not idx[b] in used_det) and detb[idx[b]].c == box.c):
                    found = True
                    tp += 1
                    used_det.append(idx[b])
                    break
            if not found:
                fn += 1
        fp = len(detb) - len(used_det)

        return tp, tn, fp, fn

    def coco(self) -> float:
        rng = [x/100 for x in range(101)]
        data = [self.calc(x) for x in rng]
        return sum(data)/len(data)

    def pascal(self) -> float:
        return self.calc(0.5)
  
@dataclass
class IOUVal:
    intersection:float =0
    union:float =0

    def __add__(self, val2:"IOUVal"):
        return IOUVal(self.intersection+val2.intersection, self.union+val2.union)
    

class mIOUAccumulator:
    def __init__(self, num_cls=21):
        self.num_cls= num_cls
        self.values = [IOUVal() for f in range(num_cls)]
    def acculumate(self,pred:Segmentation, gt:Segmentation):
        gt1 = pred.groundTruth
        gt2 = gt.groundTruth


        if (gt1.shape != gt2.shape):
            if gt1.view(-1).shape[0] < gt2.view(-1).shape[0]:
                gt2 = torch.nn.functional.interpolate(gt2.unsqueeze(0), gt1.shape[1:]).squeeze(0)
            else:
                gt1 = torch.nn.functional.interpolate(gt1.unsqueeze(0), gt2.shape[1:]).squeeze(0)
        for cls in range(self.num_cls):
            surface1 = gt1.view(-1)==cls
            surface2 = gt2.view(-1)==cls
            intersection = torch.logical_and(surface1 , surface2).sum()
            union = torch.logical_or(surface1 , surface2).sum()
            self.values[cls] += IOUVal(intersection,union)
    def val(self ):
        ious=[]
        for val in self.values:
            if val.union >0:
                ious.append(val.intersection/val.union)
        return float(torch.tensor(ious).float().mean().item())
class mIOU:
    def __init__(self, gt: List[Segmentation], val: List[Segmentation], num_cls=21, verbose = False):
        self.num_cls= num_cls
        self.gt = gt
        self.val = val
        self.filter = None
        self.verbose = verbose
        if len(gt) != len(val):
            raise Exception("Lenght must match")
    def __call__(self) -> float:
        return self.calc()
    def calc(self) -> float:
        acc = mIOUAccumulator(self.num_cls)

        for gt,val in zip(self.gt,self.val):
            acc.acculumate(val,gt)
        
        return acc.val()

        
