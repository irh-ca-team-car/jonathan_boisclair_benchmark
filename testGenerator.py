from interface.ITI import ITI
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets.detection.A2 import A2Detection
from tqdm import tqdm
import random
import math
from interface.transforms.Scale import ScaleTransform
from interface.transforms import FLIR_FIX
dataset = A2Detection("data/FLIR_CONVERTED/all.csv")

device = "cuda:0"
#m_name="unet+tu-tinynet_a"
m_name="unet+mit_b2_3->1"
model_1 = ITI.named(m_name)().to(device)

scale = ScaleTransform(224,224)

for name,model_ctr in [(m_name, model_1)]:
    model = model_ctr.to(device) #Segmenter.named("deeplabv3_resnet50")
    try:
        model.load_state_dict(torch.load(name+".pth", map_location=device), strict=False)
    except:
        pass
    optim: torch.optim.Optimizer = model.optimizer(model,2e-3,0) 
    # Assuming optimizer has two groups.
    lambda_group1 = lambda epoch: (5+(random.random()*2-1)) *10** (-(6+float(epoch) / 1000 - float(epoch) // 1000))
    lambda_group2 = lambda epoch: (5+(random.random()*2-1)) *10** (-(10+float(epoch) / 1000- float(epoch) // 1000))

    #scheduler = LambdaLR(optim, lr_lambda=[lambda_group1, lambda_group2])
    for i in range(400):
        if i>=-1:
            model.unfreeze_backbone()
        else:
            model.freeze_backbone()
        b_size = min(math.ceil(float(i+201)/25.0),24)
        batch=Batch.of(dataset,b_size)
        iter=0
        loss_fn = torch.nn.HuberLoss().to(device)
        for cocoSamp in tqdm(batch):
            cocoSamp = scale(FLIR_FIX(cocoSamp))
            cocoSamp = [c.to(device) for c in cocoSamp]
            optim.param_groups[0]["lr"] = lambda_group1(iter)
            optim.param_groups[1]["lr"] = lambda_group2(iter)
           

            optim.zero_grad()

            output = model.forward(cocoSamp)
            
            loss = sum([loss_fn(a.getGray(),b.getThermal()) for a,b in zip(output, cocoSamp)])
            loss.backward()
            Sample.show(cocoSamp[0].getRGB(), wait=False, name="gt_"+str(0))
            gray_ = output[0].getGray()

            gray_ = (gray_ - gray_.min()) / (gray_.max()-gray_.min())
            
            need_exit = (Sample.show(gray_.cpu(), wait=False, name=name)) == 27
            optim.step()

            optim.zero_grad()

           
            iter+= len(cocoSamp)
            #scheduler.step()
            if(need_exit):
                break

        torch.save(model.state_dict(), name+".pth")
        if(need_exit):
            break


# for name,model_ctr in [("unet+mit_b0", model_1)]:
#     model = model_ctr.to(device) #Segmenter.named("deeplabv3_resnet50")

#     with torch.no_grad():
#         batch=Batch.of(dataset,1)
#         for cocoSamp in tqdm(batch):
#             cocoSamp = scale(cocoSamp)
#             prediction =[f.filter(0.8) for f in model.forward(cocoSamp)]
#             Sample.show(prediction[0].onImage(cocoSamp[0]), wait=False, name=name)
#             del prediction
#             del cocoSamp