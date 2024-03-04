from .DetectionDataset import DetectionDataset
from .A1 import A1Detection
from .A2 import A2Detection
from .Citiscapes import CitiscapesDetection
from .Coco import CocoDetection
#from .VoC2007 import VoC2007Detection
from .A2W import A2W

DetectionDataset.register("coco-empty",CocoDetection())
#DetectionDataset.register("voc-2007",VoC2007Detection())