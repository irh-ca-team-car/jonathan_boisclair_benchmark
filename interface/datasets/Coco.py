from typing import Any, Tuple
from .coco.CocoDetection import CocoDetection as CD
from . import Sample
from ..detectors.Detection import Detection,Box2d
import torchvision.transforms
class CocoDetection:
    def __init__(self, root, annFile) -> None:
        self.CD = CD(root,annFile=annFile,transform=torchvision.transforms.ToTensor())
        pass
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img,ann = self.CD.__getitem__(index)
        det = Detection()
        for b in ann:
            box = Box2d()
            box.x = b["bbox"][0]
            box.y = b["bbox"][1]
            box.w = b["bbox"][2]
            box.h = b["bbox"][3]
            box.x = b["category_id"]
            det.boxes2d.append(box)
        cocoSamp = Sample()
        cocoSamp.setImage(img)
        cocoSamp.setTarget(det)
        return cocoSamp
        
