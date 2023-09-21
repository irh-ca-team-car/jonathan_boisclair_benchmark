import fiftyone
from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.transforms.Scale import ScaleTransform
import time

scale = ScaleTransform(224,224)

device = "cuda:0"
img = scale(Sample.Example().to(device))

for name,model_ctr in Segmenter.registered_Segmenters.items():
    try:
        model = model_ctr().to(device)

        prediction = model(img)

        begin = time.time()

        with torch.no_grad():
            for x in range(30):
                prediction = model(img)
        end = time.time()

        print('✓' if end-begin <1  else "X",name,":", end-begin,"s")
    except:
        pass