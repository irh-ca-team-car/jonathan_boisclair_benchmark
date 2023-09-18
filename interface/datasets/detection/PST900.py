from typing import Any, List, Tuple
from .. import Sample, LidarSample, Segmentation
from .DetectionDataset import DetectionDataset
from ...detectors.Detection import Detection, Box2d,Box3d
import torchvision.transforms
import os
import json
import torchvision.io
import numpy as np
import torch
class PSTGroup:
    def __init__(self):
        self.thermal = None
        self.rgb = None
        self.labels = None
        self.depth=None
        pass

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

class PST900Detection(DetectionDataset):
    CitiscapesClasses = ["void","safety device","backback","tool","person"]
    def classesList(self):
        return list(PST900Detection.CitiscapesClasses)
    def getId(self,str:str):
        import sys
        if str == "train":
            return self.getId("on rails")
        if str in PST900Detection.CitiscapesClasses:
            return PST900Detection.CitiscapesClasses.index(str)
        else:
            if "group" in str:
                return self.getId(str.replace("group",""))
            print(str,"is not a known category from citiscapes",file=sys.stderr)
            
            return self.getId("void")
    def getName(self,id=None):
        if id is None:
            return "Citiscapes"
        if id>=0 and id < len(PST900Detection.CitiscapesClasses):
            return PST900Detection.CitiscapesClasses[id]
        return "void"
    def isBanned(self,nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId in PST900Detection.NoTrainClass
        else:
            return self.isBanned(self.getName(nameOrId))

    def withMax(self,max) -> "PST900Detection":
        coco = PST900Detection()
        coco.images = self.images[:max]
        return coco
    def withSkip(self,maxValue) -> "PST900Detection":
        coco = PST900Detection()
        coco.images = self.images[maxValue:]
        return coco
    def shuffled(self) -> "PST900Detection":
        import random
        coco = PST900Detection()
        coco.images = [x for x in self.images]
        random.shuffle( coco.images )
        return coco

    images: List[PSTGroup]
    def __init__(self, root=None, mode="train") -> None:
        self.images = []
        images=dict()
        jsonFiles: List[Tuple(str, str)] = []
        imagesFiles: List[str] = []
        if root is None:
            root = "/media/boiscljo/LinuxData/Datasets/PST900"

        rgbs= os.listdir(os.path.join(root,mode,"rgb"))
        thermals= os.listdir(os.path.join(root,mode,"thermal_raw"))
        labels= os.listdir(os.path.join(root,mode,"labels"))
        depths = os.listdir(os.path.join(root,mode,"depth"))


        self.images = []

        for rgb,ther,lbl,dpt in zip(rgbs,thermals,labels,depths):
            group = PSTGroup()
            group.rgb = os.path.join(root,mode,"rgb",rgb)
            group.thermal = os.path.join(root,mode,"thermal_raw",ther)
            group.labels = os.path.join(root,mode,"labels",lbl)
            group.depth = os.path.join(root,mode,"depth",dpt)
            self.images.append(group)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(int(index.start),int(index.stop),int(index.step))]
            else:
                values = [v for v in range(int(index.start),int(index.stop))]
            return [self.__getitem__(v) for v in values]
        group = self.images[index]

        rgb = torchvision.io.read_image(group.rgb, torchvision.io.ImageReadMode.RGB).float()/255.0
        import cv2
        thermal = torch.from_numpy(cv2.imread(group.thermal, cv2.IMREAD_UNCHANGED).astype(np.float32)/(2**16)).unsqueeze(0)
        labels = cv2.imread(group.labels, cv2.IMREAD_UNCHANGED)

        import open3d as o3d
        
        depths = o3d.io.read_image(group.depth)
        color = o3d.io.read_image(group.rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depths)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pcl_xyz=(torch.tensor(np.asarray(pcd.points)))
        pcl_rgb=(torch.tensor(np.asarray(pcd.colors)))
        x = pcl_xyz[:,0:1]
        y = pcl_xyz[:,1:2]
        z = pcl_xyz[:,2:3]
        r = pcl_rgb[:,0:1]
        g = pcl_rgb[:,1:2]
        b = pcl_rgb[:,2:3]

        sample = Sample()
        sample._lidar = LidarSample.fromXYZIRingRGBAT(x,z,y, None,None,r,g,b,None,None)

        sample.setImage(rgb)
        sample.setThermal(thermal)

        sample._segmentation = Segmentation.FromImage(labels, self.classesList())
        sample._detection = sample._segmentation.detection
        for box in sample._detection.boxes2d:
            box.cn = self.getName(box.c)
        sample._detection.boxes2d = [box for box in sample._detection.boxes2d if box.c >0]
        return sample

