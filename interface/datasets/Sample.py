import torch
import torchvision
import torch.nn as nn

class Sample:
    img : torch.Tensor
    def __init__(self) -> None:
        self.img = torchvision.io.read_image("data/1.jpg", torchvision.io.ImageReadMode.RGB).float()/255.0
        self.img=torch.nn.functional.interpolate(self.img.unsqueeze(0) ,size=(640,640)).squeeze(0)
        self.detection = None
        #self.img = torch.zeros(3,640,640)
        pass
    def hasImage(self) -> bool:
        return True
    def isRgb(self) -> bool:
        return True
    def isArgb(self) -> bool:
        return False
    def isGray(self) -> bool:
        return False
    def getImage(self) -> torch.Tensor:
        return self.img
    def getRGB(self) -> torch.Tensor:
        if self.isGray():
            img = self.getImage()
            return torch.cat([img,img,img],0)
        elif self.isRgb():
            return self.getImage()
        elif self.isArgb():
            return self.getImage()[1:4,:,:]
    def getARGB(self) -> torch.Tensor:
        if self.isGray():
            img = self.getImage()
            return torch.cat([torch.ones(img.shape),img,img,img],0)
        elif self.isRgb():
            img = self.getImage()
            return torch.cat([torch.ones((1,*img.shape[1:])),img],0)
        elif self.isArgb():
            return self.getImage()
    def getGray(self) -> torch.Tensor:
        if self.isGray():
            return self.getImage()
        else:
            img = self.getImage()
            return torch.mean(img,0).unsqueeze(0)

    def hasLidar(self) -> bool:
        return False
    def getLidar(self) -> torch.Tensor:
        if self.hasLidar():
            return torch.zeros(300000,5)
        return None
    def hasThermal(self) -> bool:
        return False
    def getThermal(self) -> torch.Tensor:
        if self.hasThermal():
            return torch.zeros(1,640,640)
        return None
    def toTorchVisionTarget(self, device):
        if self.detection is not None:
            return self.detection.toTorchVisionTarget(device)
        return None
    def setTarget(self,detection):
        self.detection = detection