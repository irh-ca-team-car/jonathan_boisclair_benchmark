from interface.ITI import ITI
from interface.datasets.A2 import A2Detection
import torch
import cv2

from interface.datasets.Sample import Sample

def show(t: torch.Tensor,wait: bool = False):
    if len(t.shape) ==3:
        t=t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, scale_factor=(1.0,1.0))
    if len(t.shape) ==4:
        t=t[0]
    t = t.cpu().permute(1, 2, 0)
    np_ = t.detach().numpy()
    np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", np_)
    # for i in range(30):

    if wait:
        while True:
            cv2.imshow("Image", np_)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break
    else:
        cv2.waitKey(1)
device ="cuda:0" 
#for (name,iti) in ITI.allRegistered().items():
for iti in [ITI.named("DenseFuse")]:
    model :ITI = iti().to(device)
    dataset = A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.HuberLoss().to(device)

    for e in range(100):
        for sample in dataset:
            sample.to(device)
            optimizer.zero_grad()
            output :Sample = model.forward(sample)
            loss=model.loss(sample, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            show(output.getRGB())
