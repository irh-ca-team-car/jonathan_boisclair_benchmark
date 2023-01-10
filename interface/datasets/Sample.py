import torch
import torchvision
import torch.nn as nn
import numpy as np
import fiftyone as fo
class Size:
    def __init__(self,w,h) -> None:
        self.w=w
        self.h=h
    def __repr__(self) -> str:
        return "["+str(self.w)+"x"+str(self.h)+"]"
class Sample:
    img : torch.Tensor
    _thermal : torch.Tensor
    _lidar : torch.Tensor
    detection : None
   
    def Example():
        s = Sample()
        img = torchvision.io.read_image("data/1.jpg", torchvision.io.ImageReadMode.UNCHANGED).float()/255.0
        img=torch.nn.functional.interpolate(img.unsqueeze(0) ,size=(640,640)).squeeze(0)
        s.setImage(img)
        return s
    def fromFiftyOne(fiftyoneSample: fo.Sample):
        s = Sample()
        dict = fiftyoneSample.to_dict()
        img = torchvision.io.read_image(dict["filepath"], torchvision.io.ImageReadMode.UNCHANGED).float()/255.0
        s.setImage(img)
        return s
    def size(self) ->Size:
        shape = self.getRGB().shape[1:]
        return Size(shape[1],shape[0])
    def __init__(self) -> None:
        
        self.detection = None
        self._img = None
        self._thermal = None
        self._lidar = None
        #self.img = torch.zeros(3,640,640)
        pass
    def to(self,device):
        if self.img is not None:
            self.img = self.img.to(device)
        if self._thermal is not None:
            self._thermal = self._thermal.to(device)
        if self._lidar is not None:
            self._lidar = self._lidar.to(device)
        return self
    def clone(self):
        newSample = Sample()
        if self.img is not None:
            newSample.img = self.img.clone()
        if self._thermal is not None:
            newSample._thermal = self._thermal.clone()
        if self._lidar is not None:
            newSample._lidar = self._lidar.clone()
        if self.detection is not None:
            newSample.detection = self.detection.scale()
        return newSample
    def scale(self, x=1.0,y=None):
        if isinstance(x, Size):
            xFactor = x.w/self.img.shape[2]
            yFactor = x.h/self.img.shape[1]
        if y is None:
            y = x
        newSample = self
        if self.img is not None:
            img = newSample.img.unsqueeze(0)
            if not isinstance(x, Size):
                self.img=torch.nn.functional.interpolate(img,scale_factor=(y,x))[0]
            else:
                self.img=torch.nn.functional.interpolate(img,size=(x.h,x.w))[0]

        if self._thermal is not None:
            img = newSample._thermal.unsqueeze(0)
            if not isinstance(x, Size):
                self._thermal=torch.nn.functional.interpolate(img,scale_factor=(y,x))[0]
            else:
                self._thermal=torch.nn.functional.interpolate(img,size=(x.h,x.w))[0]
        if self.detection is not None:
            if not isinstance(x, Size):
                newSample.detection = self.detection.scale(x,y)
            else:
                newSample.detection = self.detection.scale(xFactor,yFactor)

        return newSample

    def setImage(self,img):
        if isinstance(img,np.ndarray):
            self.img = torch.from_numpy(img)
        if isinstance(img,torch.Tensor):
            self.img=img
        pass
    def setThermal(self,img):
        if isinstance(img,np.ndarray):
            self._thermal = torch.from_numpy(img)
        if isinstance(img,torch.Tensor):
            self._thermal=img
        pass
    def hasImage(self) -> bool:
        return self.img is not None
    def isRgb(self) -> bool:
        return self.hasImage() and self.img.shape[0]==3
    def isArgb(self) -> bool:
        return self.hasImage() and self.img.shape[0]==4
    def isGray(self) -> bool:
        return self.hasImage() and self.img.shape[0]==1
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
        return self._lidar is not None
    def getLidar(self) -> torch.Tensor:
        if self.hasLidar():
            return self._lidar
        return None
    def hasThermal(self) -> bool:
        return self._thermal is not None
    def getThermal(self) -> torch.Tensor:
        if self.hasThermal():
            return self._thermal
        return None
    def toTorchVisionTarget(self, device):
        if self.detection is not None:
            return self.detection.toTorchVisionTarget(device)
        return None
    def setTarget(self,detection):
        self.detection = detection
    
