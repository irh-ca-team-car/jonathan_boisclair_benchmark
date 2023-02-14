from typing import List, Literal, Union
from .. import Sample, LidarSample
from ...detectors.Detection import Detection, Box2d, Box3d
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
import open3d as o3d

class KittiMultiviewDetection:
    A1Classes = None
    images: fo.Dataset
    data : List[fo.Sample]

    def withMax(self,max) -> "KittiMultiviewDetection":
        coco = KittiMultiviewDetection()
        coco.data = self.data[:max]
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco
    def withSkip(self,maxValue) -> "KittiMultiviewDetection":
        coco = KittiMultiviewDetection()
        coco.data = self.data[maxValue:]
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco
    def shuffled(self) -> "KittiMultiviewDetection":
        import random
        coco = KittiMultiviewDetection()
        coco.data = [x for x in self.data]
        random.shuffle( coco.data )
        coco.images = self.images
        coco.A1Classes = self.A1Classes
        return coco

    def __init__(self, split: Union[Literal["train"],Literal["test"], None]=None,dataset_dir=None, **kwargs) -> None:
        kwargs["dataset_dir"]=dataset_dir
        
        self.images = foz.load_zoo_dataset("kitti-multiview", split=split, **kwargs)
        self.dataset_dir=dataset_dir
        # The directory containing the dataset to import
        self.dataset_dir =dataset_dir

        # Import the dataset
        dataset = self.images

        #print(dataset.group_media_types)
        #for group in (dataset.iter_groups()):
        #    print(group)
        #exit(0)

        self.n = None
        self.split = split
        self.A1Classes = None
        self.data = None

    def classesList(self):
        if self.A1Classes is None:
            #self.A1Classes =["void", *self.images.get_classes("ground_truth")]
            self.A1Classes = ["void",*self.images.distinct(
                "ground_truth.detections.label"
            )]
        return list(self.A1Classes)
    def lazy(self):
        if self.data is None:
            self.data = list(self.images)
        return self
    def getId(self,str: str):
        import sys
        if self.A1Classes is None:
            self.classesList()
        if str in self.A1Classes:
            return self.A1Classes.index(str)
        else:
            #print(str, "is not a known category from OpenImages", file=sys.stderr)
            return self.getId("void")

    def getName(self,id=None):
        if self.A1Classes is None:
            self.classesList()
        if id is None:
            return "Kitti-Multi"
        if id >= 0 and id < len(self.A1Classes):
            return self.A1Classes[id]
        return "void"

    def isBanned(self,nameOrId):
        if self.A1Classes is None:
            self.classesList()
        if isinstance(nameOrId, str):
            return nameOrId == "void" or nameOrId == "DontCare"
        else:
            return self.isBanned(self.getName(nameOrId))

   
    def __len__(self):
        return len(self.lazy().data)

    def __getitem__(self, index: int) -> Sample:

        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            values = [v for v in values if v < len(self.images)]
            if len(values)==0:
                raise StopIteration
            return [self.__getitem__(v) for v in values]
        else:
            value = self.lazy().data[index]

            citiSamp = Sample.fromFiftyOne(value)
            
            dict = value.to_dict()

            citiSamp.detection = Detection()
            if "ground_truth" in dict:
                for d in dict["ground_truth"]["detections"]:
                    box = Box2d()
                    box.x = d["bounding_box"][0] * citiSamp.getRGB().shape[2]
                    box.y = d["bounding_box"][1] * citiSamp.getRGB().shape[1]
                    box.w = d["bounding_box"][2] * citiSamp.getRGB().shape[2]
                    box.h = d["bounding_box"][3] * citiSamp.getRGB().shape[1]
                    box.c = self.getId(d["label"])
                    box.cn = d["label"]
                    if not self.isBanned(d["label"]):
                        citiSamp.detection.boxes2d.append(box)
            import pathlib

            group = self.images.get_group(value.group.id)
            pcd:fo.Sample = group["pcd"]
            dict = pcd.to_dict()
            pcdPath :str = pcd.filepath
            pcd = o3d.io.read_point_cloud(pcdPath)
            pcl_xyz=(torch.tensor(np.asarray(pcd.points)))
            pcl_rgb=(torch.tensor(np.asarray(pcd.colors)))
            x = pcl_xyz[:,0:1]
            y = pcl_xyz[:,1:2]
            z = pcl_xyz[:,2:3]
            r = pcl_rgb[:,0:1]
            g = pcl_rgb[:,1:2]
            b = pcl_rgb[:,2:3]
            citiSamp._lidar = LidarSample.fromXYZIRingRGBAT(x,y,z, None,None,r,g,b,None,None)
            
            if "ground_truth" in dict:
                for d in dict["ground_truth"]["detections"]:
                    box = Box3d()

                    _3d = d

                    box.c = self.getId(d["label"])
                    box.cn = d["label"]

                    center = _3d["location"]
                    rotation = _3d["rotation"]
                    size = _3d["dimensions"]
                    box.RPY(torch.tensor(rotation))
                    box.size = torch.tensor(size)
                    box.center = torch.tensor(center)
                    from scipy.spatial.transform import Rotation
                    q = np.array(rotation)
                    rotation = Rotation.from_euler('xyz',q)
                    w=size[0]/2
                    h=size[1]/2
                    z=size[2]/2
                    vectors = np.array([
                        [-w,h,z],
                        [w,-h,z],
                        [w,h,-z],
                        [-w,-h,z],
                        [w,-h,-z],
                        [-w,h,-z],
                        [w,h,z],
                        [-w,-h,-z]
                    ])
                    rotated_vectors = rotation.apply(vectors)
                    rotated_vectors[:,0] += center[0]
                    rotated_vectors[:,1] += center[1]
                    rotated_vectors[:,2] += center[2]

                    minx = rotated_vectors[:,0].min()
                    maxx = rotated_vectors[:,0].max()

                    miny = rotated_vectors[:,1].min()
                    maxy = rotated_vectors[:,1].max()

                    minz = rotated_vectors[:,2].min()
                    maxz = rotated_vectors[:,2].max()

                    box.x = -maxy
                    box.y = minz
                    box.z = minx

                    box.w = maxy-miny
                    box.h = maxz-minz
                    box.d = maxx-minx
                    if not self.isBanned(d["label"]):
                        citiSamp.detection.boxes3d.append(box)

            return citiSamp

