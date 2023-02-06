from typing import Callable, Dict, List, Set, Tuple
from ..detectors.Detection import Detection
import torch
from torchvision.ops import box_area, box_convert, box_iou


class MultiImageAveragePrecision:
    filter: Callable[[int],bool]
    def __init__(self, gt: List[Detection], val: List[Detection]):
        self.gt = gt
        self.val = val
        self.filter = None
        if len(gt) != len(val):
            raise Exception("Lenght must match")
    def __call__(self, iou:float) -> float:
        return self.calc(iou)
    def calc(self, iou: float) -> Dict[int,float]:
        keys: Set[int] = set()
        if self.filter is not None:
            for i in range(len(self.gt)): 
                gt = self.gt[i]
                det = self.val[i]
                gt.boxes2d = [box for box in gt.boxes2d if self.filter(box.c)]
                det.boxes2d = [box for box in gt.boxes2d if self.filter(box.c)]

        for i in range(len(self.gt)):
            gt = self.gt[i]
            det = self.val[i]
            for d in gt.boxes2d:
                keys.add(d.c)
            for d in det.boxes2d:
                keys.add(d.c)
       
        ret = dict()

        for key in keys:
            pairs = []
            for i in range(len(self.gt)):
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
            key = keys[0]
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
