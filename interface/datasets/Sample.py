import math
from typing import Dict, List, Tuple, Union
import torch
import torchvision
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import fiftyone as fo
    from .classification import ClassificationDataset
    from .detection import DetectionDataset

class Size:
    def __init__(self,w,h) -> None:
        self.w=w
        self.h=h
    @staticmethod
    def fromTensor(tensor:torch.Tensor) -> "Size":
        if len(tensor.shape) >3:
            return Size.fromTensor(tensor[0][0])
        if len(tensor.shape) >2:
            return Size.fromTensor(tensor[0])
        shape = tensor.shape
        return Size(shape[1],shape[0])
    def __repr__(self) -> str:
        return "["+str(self.w)+"x"+str(self.h)+"]"
    def div(self,value):
        return Size(self.w/value,self.h/value)
    def __div__(self,value):
        return self.div(value)
    def __mul__(self,value):
        return Size(self.w*value,self.h*value)

    def __getitem__(self, x):
        if x ==0:
            return self.w
        if x ==1:
            return self.h
        return self.__dict__[x]

    def __iter__(self):
        return iter(self.__dict__.values())

    def __len__(self):
        return 2

class LidarSample:
    # stored in X,Y,Z,I,Ring, R,G,B,A,T
    _lidar : torch.Tensor
    def clone(self) -> "LidarSample":
        copy = LidarSample()
        copy._lidar = self._lidar.clone()
        return copy
    def to(self,device) -> "LidarSample":
        self._lidar = self._lidar.to(device)
        return self
    def fromXYZ(XYZ : torch.Tensor) -> "LidarSample":
        X = XYZ[0:1,:]
        Y = XYZ[1:2,:]
        Z = XYZ[2:3,:]

        return LidarSample.fromXYZIRingRGBAT(X,Y,Z,None,None,None,None,None,None,None)
    def fromXYZI(XYZ : torch.Tensor) -> "LidarSample":
        X = XYZ[0:1,:]
        Y = XYZ[1:2,:]
        Z = XYZ[2:3,:]
        I = XYZ[3:4,:]

        return LidarSample.fromXYZIRingRGBAT(X,Y,Z,I,None,None,None,None,None,None)
    def fromXYZIRing(XYZ : torch.Tensor) -> "LidarSample":
        X = XYZ[0:1,:]
        Y = XYZ[1:2,:]
        Z = XYZ[2:3,:]
        I = XYZ[3:4,:]
        Ring = XYZ[4:5,:]

        return LidarSample.fromXYZIRingRGBAT(X,Y,Z,I,Ring,None,None,None,None,None)

    def fromXYZIRingRGBAT(X : torch.Tensor,Y : torch.Tensor,Z : torch.Tensor,I : torch.Tensor,Ring : torch.Tensor,R : torch.Tensor,G : torch.Tensor,B : torch.Tensor, A : torch.Tensor, T : torch.Tensor) -> "LidarSample":
        zeros = torch.zeros(X.shape[0],1)
        if I is None:
            I = zeros
        if Ring is None:
            Ring = zeros
        if R is None:
            R = zeros
        if G is None:
            G = zeros
        if B is None:
            B = zeros
        if A is None:
            A = zeros
        if T is None:
            T = zeros

        sample = LidarSample()
        sample._lidar = torch.cat([X,Y,Z,I,Ring,R,G,B,A,T],1)
        return sample

    def XYZ(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        return torch.cat([X,Y,Z],1)
    def ZYX(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        return torch.cat([Z,Y,X],1)
    def XYZI(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        I = self._lidar[:,3:4]
        return torch.cat([X,Y,Z,I],1)
    def IR(self) -> torch.Tensor:
        I = self._lidar[:,3:4]
        R = self._lidar[:,4:5]
        return torch.cat([I,R],1)
    def XYZIR(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        I = self._lidar[:,3:4]
        R = self._lidar[:,4:5]
        return torch.cat([X,Y,Z,I,R],1)
    def XYZRGB(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        R = self._lidar[:,5:6]
        G = self._lidar[:,6:7]
        B = self._lidar[:,7:8]
        return torch.cat([X,Y,Z,R,G,B],1)
    def RGB(self) -> torch.Tensor:
        R = self._lidar[:,5:6]
        G = self._lidar[:,6:7]
        B = self._lidar[:,7:8]
        return torch.cat([R,G,B],1)
    def XYZRGBA(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        R = self._lidar[:,5:6]
        G = self._lidar[:,6:7]
        B = self._lidar[:,7:8]
        A = self._lidar[:,8:9]
        return torch.cat([X,Y,Z,R,G,B,A],1)
    def XYZRGBPacked(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        R = (self._lidar[:,5:6] * 256).byte().float()
        G = (self._lidar[:,6:7] * 256).byte().float()
        B = (self._lidar[:,7:8] * 256).byte().float()
        RGB = R *256 *256 + G*256 + B
        return torch.cat([X,Y,Z,RGB],1)
    def XYZRGBAT(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        R = self._lidar[:,5:6]
        G = self._lidar[:,6:7]
        B = self._lidar[:,7:8]
        A = self._lidar[:,8:9]
        T = self._lidar[:,9:10]
        return torch.cat([X,Y,Z,R,G,B,A,T],1)
    def XYZRGBT(self) -> torch.Tensor:
        X = self._lidar[:,0:1]
        Y = self._lidar[:,1:2]
        Z = self._lidar[:,2:3]
        R = self._lidar[:,5:6]
        G = self._lidar[:,6:7]
        B = self._lidar[:,7:8]
        T = self._lidar[:,9:10]
        return torch.cat([X,Y,Z,R,G,B,T],1)
    def view(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.ZYX().detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.RGB().detach().cpu().numpy())
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[0, 0,0],
                                        up=[1,0,0])
    def birdseye(self,sample : "Sample",scale=25):
        xyzrgb = self.XYZRGBT()
        x = xyzrgb[:,0]
        y = xyzrgb[:,1]
        z = xyzrgb[:,2]

        if torch.nonzero(xyzrgb[:,3:6]).shape[0] ==0:
            if torch.nonzero(xyzrgb[:,6]).shape[0] != 0:
                xyzrgb[:,3]=xyzrgb[:,6]
                xyzrgb[:,4]=xyzrgb[:,6]
                xyzrgb[:,5]=xyzrgb[:,6]
            else:
                ir = self.IR()
                if torch.nonzero(ir[:,0]).shape[0] != 0:
                    xyzrgb[:,3]=ir[:,0]
                    xyzrgb[:,4]=ir[:,0]
                    xyzrgb[:,5]=ir[:,0]
                elif torch.nonzero(ir[:,1]).shape[0] != 0:
                    xyzrgb[:,3]=ir[:,1]
                    xyzrgb[:,4]=ir[:,1]
                    xyzrgb[:,5]=ir[:,1]
                else:
                    xyzrgb[:,3:6]=1

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        import math

        hw = max(abs(math.ceil(xmax)),abs(math.floor(xmin))) * scale
        width = 2*hw
        #width = (math.ceil(xmax) - math.floor(xmin)) *scale
        hh = max(abs(math.ceil(ymax)),abs(math.floor(ymin))) * scale
        height=2*hh
        #height = (math.ceil(ymax) - math.floor(ymin)) *scale

        #cx = abs(math.floor(xmin))  *scale
        #cy = abs(math.floor(ymin))  *scale
        cx = hw
        cy = hh

        cwh = torch.zeros([3,width,height]).float()

        x = (x *scale).long() + cx
        y = (y *scale).long() + cy

        cwh[:,x,y] = xyzrgb[:,3:6].permute(1,0).float()
        samp = Sample()

        cwh[:,cx,cy]= torch.tensor([0,1,0]).float()
        #cwh[1,(cx-scale):(cx+scale),(cy-scale):(cy+scale)]= 1
        samp.setImage(cwh)

        if sample.detection is not None:
            samp.detection = Detection()
            for box3d in sample.detection.boxes3d:
                box2d = Box2d()
                box2d.c = box3d.c
                box2d.x = (box3d.x * scale) + hh
                box2d.y = (box3d.y * scale) + hw
                box2d.w = box3d.w * scale
                box2d.h = box3d.h * scale
                box2d.cf = box3d.cf
                box2d.cn = box3d.cn
                samp.detection.boxes2d.append(box2d)
        return samp
class Percentage:
    __value: float
    def __init__(self,value) -> None:
        if value<0 or value>1:
            raise Exception("Invalid value, must be 0 <= value <= 1")
        self.__value = value
    @property
    def value(self):
        return self.__value

class Segmentation:
    _img : np.ndarray
    _shapes : List["Shape2d"]
    _size : Size
    def __init__(self, _img=None, _shapes=None) -> None:
        self._img = _img
        self._shapes = [] if _shapes is None else _shapes
        self._size = None
        
    def scale(self,x=1.0,y=1.0, size=None) -> "Segmentation":
        newDet = self.clone()
        if newDet._img is not None:
            newDet._img=torch.nn.functional.interpolate(torch.from_numpy(newDet._img).unsqueeze(0).unsqueeze(0),size=(size.h,size.w))[0][0].numpy()
        if newDet._shapes is not None:
            newDet._shapes = [shape.scale(x,y) for shape in newDet._shapes]
        newDet._size = size
        return newDet
    
    @staticmethod
    def FromImage(img: torch.Tensor, classesName:List[str], confidences_img:torch.Tensor=None):
        seg = Segmentation()
        import cv2
        from ..adapters.OpenCV import CVAdapter
        labels = CVAdapter.toOpenCV1Channel(img)
        seg._img=labels
        seg._size = Size.fromTensor(seg._img)
        
        for clz in range(len(classesName)):
            if clz==0: continue
            class_blobs = (labels.astype(np.uint32) == clz).astype(np.uint8)
            contours, hierarchy =cv2.findContours(class_blobs,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                nPts = contour.shape[0]

                box = Shape2d()
                box.c = clz
                box.cn = classesName[box.c]
                box.shape.extend([Point2d(contour[f,0,0],contour[f,0,1]) for f in range(nPts)])
                box.shape.append(box.shape[0])
                if confidences_img is not None:
                    box.cf =torch.mean(
                        torch.tensor(
                        [
                            confidences_img[clz,contour[f,0,0],contour[f,0,1]]
                            for f in range(nPts)
                            if contour[f,0,0] < confidences_img.shape[1] and
                            contour[f,0,1] < confidences_img.shape[2]
                        ]
                        )).item()
                    
                seg._shapes.append(box)
        return seg
    @property
    def detection(self) -> "Detection":
        import cv2
        det = Detection() 
        for shape in self._shapes:
            array = np.array([ 
                    [[int(pt.x),int(pt.y)]] for pt in shape.shape
                ])
            x_,y_,w_,h_ = cv2.boundingRect(array)
            box = Box2d()
            box.c = shape.c
            box.cn = str(box.c)
            box.x = x_
            box.y = y_
            box.w = w_
            box.h = h_
            det.boxes2d.append(box)
        return det

    @torch.no_grad()
    def onImage(self,img:Union["Sample",torch.Tensor], alpha:Percentage=Percentage(0.5), colors=None):
        import cv2
        from ..adapters.OpenCV import CVAdapter
        import random
        adapter = CVAdapter().to(img.device)
        if not isinstance(alpha,Percentage):
            alpha=Percentage(alpha)
        alpha = alpha.value
        if isinstance(img,Sample):
            img = (img.getRGB()*255.0).byte()
        size = Size.fromTensor(img)

        img: cv2.Mat = adapter.toOpenCV(img)
        gt = self.groundTruth
        if colors is None: 
            colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(128,0,0),(0,128,0),(0,0,128),(128,255,0),(255,128,0),(255,0,128),(128,0,255),(0,128,255),(0,255,128),(128,255,255),(255,128,255),(255,255,128)]
            in_max = self._img.max()
            while in_max > len(colors):
                colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        _img = torch.from_numpy(self._img).float().unsqueeze(0).unsqueeze(0)
        _img=torch.nn.functional.interpolate(_img, (size.h,size.w)).squeeze(0).squeeze(0).numpy()
        for i,c in enumerate(colors):
            mask= _img == i
            if np.array(c).max() >0:
                img[mask] = (img[mask]*(1-alpha) + np.array(c)*alpha).astype(img.dtype)
        

        return adapter.toPytorch(img)
    @property
    def groundTruth(self) -> torch.Tensor:
        import cv2
        from ..adapters.OpenCV import CVAdapter
        import random
        adapter = CVAdapter()
        if self._img is not None:
            return adapter.toPytorch(self._img)
        else:
            if self._size is None: raise Exception("Could not produce ground thruth without knowing size")
            img = torch.zeros(self._size.h,self._size.w)
            img = adapter.toOpenCV1Channel(img)
            
            for shape in self._shapes:
                array = np.array([ 
                    [[int(pt.x),int(pt.y)]] for pt in shape.shape
                ])
                #img = cv2.fillPoly(img,array,shape.c,4)
                img = cv2.drawContours(img,[array],-1,shape.c,thickness=-1)
                #raise Exception("#TODO: Implement creation of ground thruth from polygons")
                #pass

            self._img = img
            return adapter.toPytorch(self._img)

    def colored(self,colors=None) -> torch.Tensor:
        return Segmentation.color(self.groundTruth,colors)
    def toTorchVisionTarget(self,num_class, size:Size) -> torch.Tensor:
        gt = self.groundTruth.unsqueeze(0)
        gt = torch.nn.functional.interpolate(gt, size=(size.h, size.w)).squeeze(0)
        gt_ori = gt
        #print("gt_ori",gt_ori.min(),gt_ori.max())
        gt = gt.expand(num_class,-1,-1).clone()
        
        for i in range(num_class):
            gt[i] = (gt_ori[:,:]==i).float()
        #print("gt",gt.min(),gt.max())
        return gt
    @staticmethod
    def color(input,colors=None) -> torch.Tensor:
        import random
        if input.shape[0] ==1:
            input = input.squeeze(0)
        img = torch.zeros(3,*input.shape, dtype=torch.uint8)
        if colors is None: 
            colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(128,0,0),(0,128,0),(0,0,128),(128,255,0),(255,128,0),(255,0,128),(128,0,255),(0,128,255),(0,255,128),(128,255,255),(255,128,255),(255,255,128)]
            in_max=input.max()
            while in_max > len(colors):
                colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        for i,c in enumerate(colors):
            mask= (input == i)
            if np.array(c).max() >0:
                if(img[:,mask].shape[1] > 0):
                    img[:,mask] = (torch.tensor(c).byte()).unsqueeze(1).repeat(1,img[:,mask].shape[1])
        return img

    def filter(self, th)-> "Shape2d":
        newVal = self.clone()
        newVal._shapes = [x for x in newVal._shapes if x.cf > th]
        return newVal

    def __str__(self) -> str:
        return f"Segmentation(_img={self._img},_shapes={self._shapes})"
    def __repr__(self) -> str:
        return self.__str__()
    def clone(self)-> "Segmentation":
        newValue = Segmentation()
        newValue._img= np.copy(self._img) if self._img is not None else None
        newValue._size = self._size
        newValue._shapes= [shape.clone() for shape in self._shapes] if self._shapes is not None else None
        return newValue

class Sample:
    _img : torch.Tensor
    _thermal : torch.Tensor
    _lidar : LidarSample
    _detection: "Detection"
    _classification: "Classification"
    _segmentation: "Segmentation"
    

    def __init__(self) -> None:
        self._detection = None
        self._classification = None
        self._img = None
        self._thermal = None
        self._lidar = None
        self._segmentation=None
        #self._img = torch.zeros(3,640,640)
        pass

    @staticmethod
    def Example() -> "Sample":
        s = Sample()
        img = torchvision.io.read_image("data/1.jpg", torchvision.io.ImageReadMode.UNCHANGED).float()/255.0
        img=torch.nn.functional.interpolate(img.unsqueeze(0) ,size=(640,640)).squeeze(0)
        s.setImage(img)
        s.detection = Detection()
        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[0].x = 150
        s.detection.boxes2d[0].y = 325
        s.detection.boxes2d[0].w = 135
        s.detection.boxes2d[0].h = 175
        s.detection.boxes2d[0].c = 2
        s.detection.boxes2d[0].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 344
        s.detection.boxes2d[-1].y = 353
        s.detection.boxes2d[-1].w = 80
        s.detection.boxes2d[-1].h = 80
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 429
        s.detection.boxes2d[-1].y = 346
        s.detection.boxes2d[-1].w = 50
        s.detection.boxes2d[-1].h = 65
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 471
        s.detection.boxes2d[-1].y = 346
        s.detection.boxes2d[-1].w = 30
        s.detection.boxes2d[-1].h = 65
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 515
        s.detection.boxes2d[-1].y = 305
        s.detection.boxes2d[-1].w = 10
        s.detection.boxes2d[-1].h = 10
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 503
        s.detection.boxes2d[-1].y = 310
        s.detection.boxes2d[-1].w = 10
        s.detection.boxes2d[-1].h = 10
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 491
        s.detection.boxes2d[-1].y = 313
        s.detection.boxes2d[-1].w = 10
        s.detection.boxes2d[-1].h = 10
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 485
        s.detection.boxes2d[-1].y = 316
        s.detection.boxes2d[-1].w = 10
        s.detection.boxes2d[-1].h = 10
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 502
        s.detection.boxes2d[-1].y = 328
        s.detection.boxes2d[-1].w = 40
        s.detection.boxes2d[-1].h = 62
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 540
        s.detection.boxes2d[-1].y = 360
        s.detection.boxes2d[-1].w = 30
        s.detection.boxes2d[-1].h = 30
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 544
        s.detection.boxes2d[-1].y = 349
        s.detection.boxes2d[-1].w = 30
        s.detection.boxes2d[-1].h = 10
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 565
        s.detection.boxes2d[-1].y = 350
        s.detection.boxes2d[-1].w = 20
        s.detection.boxes2d[-1].h = 20
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 593
        s.detection.boxes2d[-1].y = 351
        s.detection.boxes2d[-1].w = 15
        s.detection.boxes2d[-1].h = 30
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 600
        s.detection.boxes2d[-1].y = 355
        s.detection.boxes2d[-1].w = 30
        s.detection.boxes2d[-1].h = 40
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        s.detection.boxes2d.append(Box2d())
        s.detection.boxes2d[-1].x = 604
        s.detection.boxes2d[-1].y = 352
        s.detection.boxes2d[-1].w = 30
        s.detection.boxes2d[-1].h = 40
        s.detection.boxes2d[-1].c = 2
        s.detection.boxes2d[-1].cn = "car"

        return s
    @property
    def device(self) -> torch.device:
        if self._img is not None: return self._img.device
        if self._thermal is not None: return self._thermal.device
        if self._lidar is not None: return self._lidar._lidar.device
        return torch.device("cpu")
    @property
    def segmentation(self) -> "Segmentation":
        return self._segmentation
    @segmentation.setter
    def segmentation(self,value) -> "Segmentation":
        self._segmentation = value
        return value
    @property
    def detection(self) -> "Detection":
        if self._detection is not None:
            return self._detection
        if self._segmentation is not None:
            return self._segmentation.detection
        return None
    @detection.setter
    def detection(self,value) -> "Detection":
        self._detection = value
        return value
    @property
    def classification(self) -> "Classification":
        return self._classification
    @classification.setter
    def classification(self,value) -> "Classification":
        self._classification = value
        return value
    
    def fromFiftyOne(fiftyoneSample: "fo.Sample") -> "Sample":
        s = Sample()
        dict = fiftyoneSample.to_dict()
        img = torchvision.io.read_image(dict["filepath"], torchvision.io.ImageReadMode.UNCHANGED).float()/255.0
        s.setImage(img)
        return s
    def size(self) ->Size:
        if self._img is not None:
            shape = self.getRGB().shape[1:]
        else: 
            thermal_ = self.getThermal()
            if len(thermal_.shape) ==2:
                shape = thermal_.shape
            else:
                shape = thermal_.shape[1:]
            
        return Size(shape[1],shape[0])

    def to(self,device) -> "Sample":
        if self._img is not None:
            self._img = self._img.to(device)
        if self._thermal is not None:
            self._thermal = self._thermal.to(device)
        if self._lidar is not None:
            self._lidar = self._lidar.to(device)
        if self.detection is not None:
            self.detection = self.detection.to(device)
        if self.classification is not None:
            self.classification = self.classification.to(device)
        return self
    def clone(self) -> "Sample":
        newSample = Sample()
        if self._img is not None:
            newSample._img = self._img.clone()
        if self._thermal is not None:
            newSample._thermal = self._thermal.clone()
        if self._lidar is not None:
            newSample._lidar = self._lidar.clone()
        if self.detection is not None:
            newSample.detection = self.detection.scale()
        if self.classification is not None:
            newSample.classification = self.classification.clone()
        if self._segmentation is not None:
            newSample._segmentation = self._segmentation.clone()
        return newSample
    def crop(self,new_x:int,new_y:int,new_width:int,new_height:int, overlap_to_keep=0.2) -> "Sample":
        newSample = self.clone()
        if self._img is not None:
            new_image = self._img[:,new_y:(new_height+new_y),new_x:(new_width+new_x)]
            newSample.setImage(new_image)
        if self._thermal is not None:
            new_image = self._thermal[:,new_y:(new_height+new_y),new_x:(new_width+new_x)]
            newSample.setThermal(new_image)
        if self.detection is not None:
            newSample.setTarget(self.detection.crop(new_x,new_y,new_width,new_height,overlap_to_keep))
        newSample._lidar = self._lidar
        
        return newSample
    def scale(self, x=1.0,y=None) -> "Sample":
        if isinstance(x, Size):
            xFactor = x.w/self.size().w#self._img.shape[2]
            yFactor = x.h/self.size().h#self._img.shape[1]
        if y is None:
            y = x
        newSample = self.clone()
        if newSample._img is not None:
            img = newSample._img.unsqueeze(0)
            if not isinstance(x, Size):
                newSample._img=torch.nn.functional.interpolate(img,scale_factor=(y,x))[0]
            else:
                newSample._img=torch.nn.functional.interpolate(img,size=(x.h,x.w))[0]
        if newSample._thermal is not None:
            img = newSample._thermal.unsqueeze(0)
            if not isinstance(x, Size):
                newSample._thermal=torch.nn.functional.interpolate(img,scale_factor=(y,x))[0]
            else:
                newSample._thermal=torch.nn.functional.interpolate(img,size=(x.h,x.w))[0]
        if newSample.detection is not None:
            if not isinstance(x, Size):
                newSample.detection = newSample.detection.scale(x,y)
            else:
                newSample.detection = newSample.detection.scale(xFactor,yFactor)
        if newSample._segmentation is not None:
            if not isinstance(x, Size):
                newSample._segmentation = newSample._segmentation.scale(x,y)
            else:
                newSample._segmentation = newSample._segmentation.scale(xFactor,yFactor, size=x)

        return newSample

    def setImage(self,img) -> "Sample":
        if isinstance(img,np.ndarray):
            self._img = torch.from_numpy(img)
        if isinstance(img,torch.Tensor):
            self._img=img
        return self
    def setThermal(self,img)-> "Sample":
        if isinstance(img,np.ndarray):
            self._thermal = torch.from_numpy(img)
        if isinstance(img,torch.Tensor):
            self._thermal=img
        return self 
    def hasImage(self) -> bool:
        return self._img is not None
    def isRgb(self) -> bool:
        return self.hasImage() and self._img.shape[0]==3
    def isArgb(self) -> bool:
        return self.hasImage() and self._img.shape[0]==4
    def isGray(self) -> bool:
        return self.hasImage() and self._img.shape[0]==1
    def getImage(self) -> torch.Tensor:
        return self._img
    def getRGB(self) -> torch.Tensor:
        if self.isGray():
            img = self.getImage()
            return torch.cat([img,img,img],0)
        elif self.isRgb():
            return self.getImage()
        elif self.isArgb():
            return self.getImage()[1:4,:,:]
    def getRGBT(self) -> torch.Tensor:
        return torch.cat([self.getRGB(),self.getThermal()])
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
    def getLidar(self) -> LidarSample:
        if self.hasLidar():
            return self._lidar
        return None
    def hasThermal(self) -> bool:
        return self._thermal is not None
    def getThermal(self) -> torch.Tensor:
        if self.hasThermal():
            return self._thermal
        return None
    def toTorchVisionTarget(self, device) -> Dict[str,torch.Tensor]:
        if self.detection is not None:
            return self.detection.toTorchVisionTarget(device)
        return None
    def toTorchVisionSegmentationTarget(self, num_class,size) -> Dict[str,torch.Tensor]:
        if self.segmentation is not None:
            return self.segmentation.toTorchVisionTarget(num_class,size)
        return None
    def setTarget(self,detection) -> "Sample":
        self.detection = detection
        return self
    
    @staticmethod
    def show(t: torch.Tensor, wait: bool = False, name="Image") -> int:
        import cv2
        if len(t.shape) == 4:
            t = t[0]
        if t.shape[0] ==3:
            t = t.cpu().permute(1, 2, 0)
            np_ = t.detach().numpy()
            np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
        elif t.shape[0] ==1:
            np_ = t[0].detach().numpy()
        
        try:
            cv2.imshow(name, np_)
            # for i in range(30):

            if wait:
                while True:
                    cv2.imshow(name, np_)
                    k = cv2.waitKey(1)
                    if k == 27:  # Esc key to stop
                        break
            else:
                return cv2.waitKey(1)
        except:
            return 0
class Box2d:
    x: float
    y: float
    w: float
    h: float
    c: float
    cf: float
    cn: str

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = 0
        self.c = 0
        self.cf = 0
        self.cn = ""
    def scale(self,x=1.0,y=1.0) -> "Box2d":
        newBox = Box2d()
        newBox.x = self.x*x
        newBox.y = self.y*y
        newBox.w = self.w*x
        newBox.h = self.h*y
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        return newBox
    def surface(self):
        return self.w*self.h

    def __str__(self) -> str:
        return f"Box2d[x:{self.x},y:{self.y},w:{self.w},h:{self.h},class:{self.c},confidence:{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()
class Point2d:
    x: float
    y: float
    def __init__(self,x=0,y=0) -> None:
        self.x=x
        self.y=y
    def __mul__(self,pt:Union[int,float,"Point2d"]) -> "Point2d":
        if not isinstance(pt,Point2d):
            return self * Point2d(float(pt),float(pt))
        return Point2d(self.x*pt.x,self.y*pt.y)
    def clone(self) -> "Point2d":
        return Point2d(self.x,self.y)
    def __str__(self) -> str:
        return f"Point2d({self.x},{self.y})"
    def __repr__(self) -> str:
        return self.__str__()
class Shape2d:
    c: float
    cf: float
    cn: str
    shape: List[Point2d]
    _surface : Union[float,None]

    def __init__(self) -> None:
        self.c = 0
        self.cf = 0
        self.cn = ""
        self.shape=[]
        self._surface=None

    def scale(self,x=1.0,y=1.0) -> "Shape2d":
        clone = self.clone()
        clone.shape = [(p * Point2d(x,y)) for p in self.shape]

        return clone
    @property
    def surface(self) -> float:
        if self._surface is not None: return self._surface
        return 0 #TODO: Calculate surface from shape
    
    def clone(self) -> "Shape2d":
        newBox = Shape2d()
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        newBox.shape = [p.clone() for p in self.shape]
        return newBox

    def __str__(self) -> str:
        return f"Shape2d[shape:{self.shape},class:{self.c},confidence:{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()
    
class Quaternion():
    def __init__(self,tensor) -> None:
        self.x = tensor[0]
        self.y = tensor[1]
        self.z = tensor[2]
        self.w = tensor[3]

class Box3d:
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float
    c: float
    cf: float
    cn: str
    quat: torch.Tensor
    center: torch.Tensor
    size: torch.Tensor
    def scale(self,x=1.0,y=1.0, z=1.0) -> "Box3d":
        newBox = Box3d()
        newBox.x = self.x*x
        newBox.y = self.y*y
        newBox.z = self.z*z
        newBox.w = self.w*x
        newBox.h = self.h*y
        newBox.d = self.d*y
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        newBox.quat = self.quat.clone()
        newBox.center = self.center.clone()
        newBox.size = self.size.clone()
        return newBox
   

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = self.z = self.d = 0
        self.c = 0
        self.cf = 0
        self.cn = ""
        self.quat = torch.tensor([0,0,0,1.0])
        self.center = torch.zeros(3)
        self.size = torch.zeros(3)
    def Quat(self, value:torch.Tensor=None)-> torch.Tensor:
        if value is not None:
            self.quat = value
        return self.quat
    def RPY(self, value:torch.Tensor=None)-> torch.Tensor:
        if value is not None:
            cr = math.cos(value[0] * 0.5)
            sr = math.sin(value[0] * 0.5)
            cp = math.cos(value[1] * 0.5)
            sp = math.sin(value[1] * 0.5)
            cy = math.cos(value[2] * 0.5)
            sy = math.sin(value[2] * 0.5)

            self.quat[3] = cr * cp * cy + sr * sp * sy
            self.quat[0] = sr * cp * cy - cr * sp * sy
            self.quat[1] = cr * sp * cy + sr * cp * sy
            self.quat[2] = cr * cp * sy - sr * sp * cy

        ret = torch.Tensor(3)
        q = Quaternion(self.quat)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        ret[0] = math.atan2(sinr_cosp, cosr_cosp)

        #pitch (y-axis rotation)
        sinp = math.sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
        cosp = math.sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
        ret[1] = 2 * math.atan2(sinp, cosp) - math.pi / 2

        #yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        ret[2] = math.atan2(siny_cosp, cosy_cosp)
        return ret

    def __str__(self) -> str:
        return f"Box3d[RPY:{self.RPY()},center:{self.center},size:{self.size},class:{self.cn}#{self.c},confidence:{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


class Detection:
    boxes2d: List[Box2d]
    boxes3d: List[Box3d]
    device : torch.device

    def __init__(self) -> None:
        self.boxes2d = []
        self.boxes3d = []
        self.device = torch.device("cpu")
        
    def scale(self,x=1.0,y=1.0) -> "Detection":
        newDet = Detection()
        newDet.boxes2d = [b.scale(x,y) for b in self.boxes2d]
        newDet.boxes3d = list([b.scale() for b in self.boxes3d])
        return newDet
    def to(self,device)-> "Detection":
        return self
    def fromTorchVision(torchVisionResult, dataset=None)-> "Detection":
        ret = []
        for res in torchVisionResult:
            det = Detection()
            for b in range(res["boxes"].shape[0]):
                box = Box2d()
                box.x = res["boxes"][b, 0].item()
                box.y = res["boxes"][b, 1].item()
                box.w = res["boxes"][b, 2].item()-res["boxes"][b, 0].item()
                box.h = res["boxes"][b, 3].item()-res["boxes"][b, 1].item()
                box.c = res["labels"][b].item()
                box.cf = res["scores"][b].item()
                box.cn = str(box.c)#CocoDetection.getName(box.c)
                if dataset is not None:
                    box.cn = dataset.getName(box.c)
                det.boxes2d.append(box)
            ret.append(det)

        if len(ret) == 1:
            return ret[0]
        if len(ret) == 0:
            return None
        return ret

    def filter(self, th)-> "Detection":
        newVal = Detection()
        newVal.device = self.device
        newVal.boxes2d = [x for x in self.boxes2d if x.cf > th]
        newVal.boxes3d = [x for x in self.boxes3d if x.cf > th]
        return newVal
    def c(self,c)-> "Detection":
        d = Detection()
        d.device = self.device
        d.boxes2d = [x for x in self.boxes2d if int(x.c) == int(c)]
        d.boxes3d = [x for x in self.boxes3d if int(x.c) == int(c)]
        return d
    @torch.no_grad()
    def onImage(self, sample:Union[Sample,torch.Tensor], colors:List[Tuple[int,int,int]]=None, width=4)->torch.Tensor:
        if isinstance(sample,Sample):
            img = (sample.getRGB()*255.0).byte()
        elif isinstance(sample,torch.Tensor):
            img = sample
            if img.dtype == torch.float32:
                img = (img*255.0).byte()
        else :
            raise Exception("Argument sample must be sample or tensor")
        img = img.to("cpu")
        target = self.toTorchVisionTarget("cpu")
        if len(self.boxes2d) > 0:
            labels = [b.cn+"@"+str(b.cf) if b.cf > 0.01 else b.cn for b in self.boxes2d]
            if colors is not None:
                colors = [c for c in colors]
                i=0
                while len(colors) < len(labels):
                    colors.append(colors[i])
                    i+=1
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=width, colors=colors)
                pass
            else:
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=width)
        return img.to(self.device)
    def toX1Y1X2Y2C(self,device=None) -> torch.Tensor:
        if device is None:
            device = self.device
        ret= torch.tensor([[x.x,x.y,x.x+x.w,x.y+x.h,x.c-1]for x in self.boxes2d]).to(device)
        if(len(ret.shape) == 1):
            ret = ret.view(0,5)
        return ret.to(device)
    def toX1Y1X2Y2CFC(self,device=None)-> torch.Tensor:
        if device is None:
            device = self.device
        ret= torch.tensor([[x.x,x.y,x.x+x.w,x.y+x.h,x.cf,x.c-1]for x in self.boxes2d]).to(device)
        if(len(ret.shape) == 1):
            ret = ret.view(0,6)
        return ret
    def toTorchVisionTarget(self, device=None) -> Dict[str,torch.Tensor]:
        if device is None:
            device = self.device
        boxes = []
        labels = []
        scores = []
        for box in self.boxes2d:
            if box.x<=0:
                box.x = 0
            if box.y<=0:
                box.y = 0
            if box.w<=1:
                box.w=2
            if box.h<=1:
                box.h=2
            boxes.append([box.x, box.y, box.x+box.w, box.y+box.h])
            if boxes[-1][2] <= boxes[-1][0]:
                boxes[-1][2] = boxes[-1][0] +1
            if boxes[-1][3] <= boxes[-1][1]:
                boxes[-1][3] = boxes[-1][1] +1
            labels.append(int(box.c))
            scores.append(int(box.cf))
        t= torch.tensor(boxes, dtype=torch.int64).to(device)
        if(len(boxes)==0):
            t = t.view(0,4)
        return {'boxes': t, 'labels': torch.tensor(labels, dtype=torch.int64).to(device), 'scores':torch.tensor(scores, dtype=torch.int64).to(device)}

    def __str__(self) -> str:
        return "{ type:Detection, boxes2d:"+str(self.boxes2d)+", boxes3d:"+str(self.boxes3d) + "}"

    def __repr__(self) -> str:
        return self.__str__()

    def NMS(self, overlapThresh = 0.4) -> "Detection" :
        import numpy as np
        newDetection = self.filter(0)
        # Return an empty list, if no boxes given
        if len(newDetection.boxes2d) == 0:
            return newDetection
        boxes = newDetection.toX1Y1X2Y2C("cpu").numpy()
        x1 = boxes[:, 0]  # x coordinate of the top-left corner
        y1 = boxes[:, 1]  # y coordinate of the top-left corner
        x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
        y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
        # Compute the area of the bounding boxes and sort the bounding
        # Boxes by the bottom-right y-coordinate of the bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
        # The indices of all boxes at start. We will redundant indices one by one.
        indices = np.arange(len(x1))
        for i,box in enumerate(boxes):
            # Create temporary indices  
            temp_indices = indices[indices!=i]
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0], boxes[temp_indices,0])
            yy1 = np.maximum(box[1], boxes[temp_indices,1])
            xx2 = np.minimum(box[2], boxes[temp_indices,2])
            yy2 = np.minimum(box[3], boxes[temp_indices,3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / areas[temp_indices]
            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
            if np.any(overlap) > overlapThresh:
                indices = indices[indices != i]
            #return only the boxes at the remaining indices
        newDetection.boxes2d = [newDetection.boxes2d[i] for i in indices]
        #newDetection.boxes2d = newDetection.boxes2d[indices].astype(int)
        newDetection.device = self.device
        return newDetection
    def NMS_Pytorch(self,thresh_iou : float=0.4)-> "Detection" :
        """
        Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the image 
                along with the class predscores, Shape: [num_boxes,5].
            thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
        Returns:
            A list of filtered boxes, Shape: [ , 5]
        """
        P = self.toX1Y1X2Y2CFC()
        newDetection = self.filter(0)
        # we extract coordinates for every 
        # prediction box present in P
        x1 = P[:, 0]
        y1 = P[:, 1]
        x2 = P[:, 2]
        y2 = P[:, 3]
    
        # we extract the confidence scores as well
        scores = P[:, 4]
        labels = P[:, 5]
    
        # calculate area of every block in P
        areas = (x2 - x1) * (y2 - y1)
        
        # sort the prediction boxes in P
        # according to their confidence scores
        order = scores.argsort()
    
        # initialise an empty list for 
        # filtered prediction boxes
        keep = []
        
    
        while len(order) > 0:
            
            # extract the index of the 
            # prediction with highest score
            # we call this prediction S
            idx = order[-1]
    
            # push S in filtered predictions list
            keep.append(idx)
    
            # remove S from P
            order = order[:-1]
    
            # sanity check
            if len(order) == 0:
                break
            
            # select coordinates of BBoxes according to 
            # the indices in order
            xx1 = torch.index_select(x1,dim = 0, index = order)
            xx2 = torch.index_select(x2,dim = 0, index = order)
            yy1 = torch.index_select(y1,dim = 0, index = order)
            yy2 = torch.index_select(y2,dim = 0, index = order)
    
            # find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])
    
            # find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1
            
            # take max with 0.0 to avoid negative w and h
            # due to non-overlapping boxes
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
    
            # find the intersection area
            inter = w*h
    
            # find the areas of BBoxes according the indices in order
            rem_areas = torch.index_select(areas, dim = 0, index = order) 
    
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            
            # find the IoU of every prediction in P with S
            IoU = inter / union
    
            # keep the boxes with IoU less than thresh_iou
            mask = IoU < thresh_iou
            order = order[mask]
        
        newDetection.boxes2d = [newDetection.boxes2d[i] for i in keep]
        #newDetection.boxes2d = newDetection.boxes2d[indices].astype(int)
        newDetection.device = self.device
        return newDetection
    def crop(self,new_x:int,new_y:int,new_width:int,new_height:int, overlap_to_keep=0.2) -> "Sample":
        newDet = Detection()

        for box2d in self.boxes2d:
            new_x_box = max(box2d.x, new_x) - new_x
            new_y_box = max(box2d.y, new_y) - new_y

            old_x2 = box2d.x + box2d.w
            old_y2 = box2d.y + box2d.h

            loss_left = new_x - box2d.x if new_x > box2d.x else 0
            loss_top = new_y - box2d.y if new_y > box2d.y else 0

            loss_right = max(0,old_x2- (new_x+new_width))
            loss_bottom= max(0,old_y2- (new_y+new_height))

            newBox = box2d.scale()
            newBox.x = new_x_box
            newBox.y= new_y_box
            newBox.w = box2d.w - loss_left - loss_right
            newBox.h = box2d.h - loss_top - loss_bottom

            if newBox.w < 0 or newBox.h < 0:
                continue
            if newBox.surface() < box2d.surface() * overlap_to_keep:
                continue
            newDet.boxes2d.append(newBox)

        newDet.boxes3d = list([b.scale() for b in self.boxes3d])
        return newDet

class Classification:
    confidences : torch.Tensor
    device: torch.device
    dataset: "ClassificationDataset"
    def __init__(self, confidences: Union[torch.Tensor,int], dataset: "ClassificationDataset" = None) -> None:
        if isinstance(confidences,int):
            self.confidences = torch.zeros(len(dataset.classesList()))
            self.confidences[confidences]=1
        else:
            self.confidences=confidences
        self.device=torch.device("cpu")
        self.dataset = dataset
    def clone(self)-> "Classification":
        return Classification(self.confidences, self.dataset).to(self.device)
    def to(self,device) -> "Classification":
        self.device = device
        self.confidences = self.confidences.to(device)
        return self
    def getConfidence(self)-> float:
        return self.confidences[self.confidences.argmax().item()].item()
    def getCategory(self)-> int:
        return self.confidences.argmax().item()
    def getCategoryName(self)->str:
        return self.dataset.getName(self.getCategory())
    def __repr__(self) -> str:
        return self.confidences.__repr__()
    def __str__(self) -> str:
        return self.confidences.__str__()