from .DetectionDataset import DetectionDataset
from .A1 import A1Detection
from .A2 import A2Detection
from .Citiscapes import CitiscapesDetection
from .Coco import CocoDetection

DetectionDataset.register("coco-empty",CocoDetection())