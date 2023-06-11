from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch

device = "cuda:0"
img = Sample.Example().to(device)
master_model = Segmenter.named("deeplabv3_resnet50").to(device)
img.segmentation = master_model.forward(img)
master_model = Segmenter.named("fcn_resnet101").to(device)

optim = torch.optim.Adamax(master_model.parameters(),lr=2e-6)
for b in range(5):
    optim.zero_grad()
    loss = (master_model.calculateLoss([img]))
    loss.backward()
    prediction =[f.filter(0.8) for f in master_model.forward([img])]
    print(loss)
    optim.step()
    Sample.show(prediction[0].onImage(img), wait=False, name="pre-train")

    optim.zero_grad()
img.segmentation = master_model.forward(img)

for name,model_ctr in Segmenter.registered_Segmenters.items():
    model = model_ctr().to(device) #Segmenter.named("deeplabv3_resnet50")

    optim = torch.optim.Adamax(model.parameters(),lr=2e-6)
    for b in range(50):
        optim.zero_grad()
        loss = (model.calculateLoss([img]))
        loss.backward()
        prediction =[f.filter(0.8) for f in model.forward([img])]
        print(loss)
        optim.step()
        Sample.show(prediction[0].onImage(img), wait=False, name=name)

        optim.zero_grad()
