import torch
import torch.nn as nn
class Sample:
    def __init__(self) -> None:
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
        return torch.zeros(3,640,640)
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
        return True
    def getLidar(self) -> torch.Tensor:
        return torch.zeros(300000,5)
    def hasThermal(self) -> bool:
        return True
    def getThermal(self) -> torch.Tensor:
        return torch.zeros(1,640,640)