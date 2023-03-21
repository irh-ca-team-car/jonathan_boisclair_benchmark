from ..datasets import Sample
import cv2
import torch
import numpy
class CVAdapter:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
    @staticmethod
    def toOpenCV(t:torch.Tensor) ->cv2.Mat:
        if len(t.shape) ==4:
            t=t[0]
        t = t.cpu().permute(1, 2, 0)
        np_ = t.detach().numpy()
        np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
        return np_
    @staticmethod
    def toOpenCV1Channel(t:torch.Tensor) ->cv2.Mat:
        import numpy as np
        if isinstance(t,np.ndarray): return t
        if len(t.shape) ==4:
            t=t[0]
        t = t.cpu()
        np_ = t.detach().numpy()
        return np_
    def to(self,device:torch.device):
        self.device = device
        return self
    def toPytorch(self,np: cv2.Mat) -> torch.Tensor:

        if np.dtype == numpy.uint16:
            return torch.tensor(np.astype(numpy.float32)).unsqueeze(0).float() / 2**16
        elif np.dtype == numpy.uint8:
            tensor = torch.from_numpy(np)
            if len(tensor.shape) == 3 and tensor.shape[2] ==3:
                b = tensor[:,:,0].unsqueeze(0)
                g = tensor[:,:,1].unsqueeze(0)
                r = tensor[:,:,2].unsqueeze(0)
                return (torch.cat([r,g,b],0).float()/255).to(self.device)
            else:
                return tensor
        elif np.dtype == numpy.float32:
            return torch.tensor(np.astype(numpy.float32))
        else:
            print("How to convert?",np.dtype)

class VideoCapture:
    def __init__(self,path) -> None:
        self.cap = cv2.VideoCapture(path)
        self.device = torch.device("cpu")
    def __iter__(self):
        return self
    def to(self,device:torch.device):
        self.device = device
        return self
    def __next__(self) -> Sample:
        s = Sample()
        try:
            ret, frame = self.cap.read()
            s = Sample()
            s.setImage(CVAdapter().to(self.device).toPytorch(frame))
            return s
        except BaseException as e:
            print(e)
            raise StopIteration(e)
    