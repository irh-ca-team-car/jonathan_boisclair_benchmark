from typing import Any, Tuple
from .DetectionDataset import DetectionDataset
from ..coco.CocoDetection import CocoDetection as CD
from .. import Sample
from ...detectors.Detection import Detection,Box2d
import torchvision.transforms

mscoco = ['__background__',
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 '',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 '',
 'backpack',
 'umbrella',
 '',
 '',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 '',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 '',
 'dining table',
 '',
 '',
 'toilet',
 '',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 '',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']
 
class CocoDetection(DetectionDataset):
    def classesList(self):
        return list(mscoco)
    def getId(self,str:str):
        import sys
        if str == "train":
            return CocoDetection.getId("on rails")
        if str in mscoco:
            return mscoco.index(str)
        else:
            if "group" in str:
                return CocoDetection.getId(str.replace("group",""))
            print(str,"is not a known category from MSCOCO",file=sys.stderr)
            
            return CocoDetection.getId("void")
    def getName(self,id=None):
        if self is not None and not isinstance(self,CocoDetection):
            return CocoDetection.getName(None,self)
        if id is None:
            return "MS-COCO"
        if id>=0 and id < len(mscoco):
            #print(id,mscoco[id])
            return mscoco[id]
        else:
            return str(id)

    def __init__(self, root=None, annFile=None) -> None:
        self.max=None
        self.min=0
        if root is None or annFile is None:
            self.CD = None
        else:
            self.CD = CD(root,annFile=annFile,transform=torchvision.transforms.ToTensor())
        pass
    def withMax(self,max):
        coco = CocoDetection()
        coco.max = max
        coco.min= self.min
        coco.CD = self.CD
        return coco
    def withSkip(self,maxValue) -> "DetectionDataset":
        coco = CocoDetection()
        coco.max = self.max
        coco.min= maxValue
        coco.CD = self.CD
        return coco
    def shuffled(self) -> "DetectionDataset":
        import random
        coco = CocoDetection()
        coco.max = self.max
        coco.min= self.min
        #coco.CD = random.shuffle( self.CD )
        return coco

    def __len__(self):
        if self.CD is not None:
            if self.max is not None:
                return min(len(self.CD)-self.min,self.max)
            return len(self.CD)
        return 0
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.CD is None:
            raise StopIteration()
        if self.max is not None and index > self.max:
            raise StopIteration()
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            return [self.__getitem__(v) for v in values]
        img,ann = self.CD.__getitem__(index)
        det = Detection()
        for b in ann:
            box = Box2d()
            box.x = b["bbox"][0]
            box.y = b["bbox"][1]
            box.w = b["bbox"][2]
            box.h = b["bbox"][3]
            box.c = b["category_id"]
            box.cn = CocoDetection.getName(box.c)
            det.boxes2d.append(box)
        cocoSamp = Sample()
        cocoSamp.setImage(img)
        cocoSamp.setTarget(det)
        return cocoSamp
        
