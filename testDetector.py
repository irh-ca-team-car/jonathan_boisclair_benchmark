import fiftyone
import fiftyone.zoo
from interface.detectors.Detector import Detector
from interface.classifiers.Classifier import Classifier
Detector.named("ssd_lite")
Classifier.named("alexnet")