from typing import Any, List, Tuple
from . import Sample
from ..detectors.Detection import Detection, Box2d,Box3d
import torchvision.transforms
import os
import json
import torchvision.io
import numpy as np

class A1Detection:
    A1Classes = ["void"	,"","","",""]
    def classesList():
        return list(A1Detection.A1Classes)
    def getId(str:str):
        import sys
        if str in A1Detection.A1Classes:
            return A1Detection.A1Classes.index(str)
        else:
            print(str,"is not a known category from A1",file=sys.stderr)
            
            return A1Detection.getId("void")
    def getName(id=None):
        if id is None:
            return "A1"
        if id>=0 and id < len(A1Detection.A1Classes):
            return A1Detection.A1Classes[id]
        return "void"
    def isBanned(nameOrId):
        if isinstance(nameOrId,str):
            return nameOrId == "void"
        else:
            return A1Detection.isBanned(A1Detection.getName(nameOrId))

    images: List[CitiscapesGroup]
    def __init__(self, root=None, mode="train", suffix="8bit.png") -> None:
        self.images = []
        images=dict()
        jsonFiles: List[Tuple(str, str)] = []
        imagesFiles: List[str] = []
        if root is None:
            root = "/media/boiscljo/LinuxData/Datasets/Citiscapes/a"
            for (dirpath, dirnames, filenames) in os.walk(root):
                for filename in filenames:
                    if filename.endswith('.json'):
                        if "Bbox" in filename or "_gtF" in filename or "_gtC" in filename:
                            if "/"+mode+"/" in dirpath:
                                jsonFiles.append((dirpath, filename))
                    if filename.endswith('.png'):
                        if "/"+mode+"/" in dirpath:
                            imagesFiles.append((dirpath, filename))

        for dirpath, filename in jsonFiles:
            id = "_".join(filename.split("_")[0:3])
            if id in images:
                obj = images[id]
            else:
                obj = images[id] = CitiscapesGroup()
            obj.id = id
            if "Bbox3d" in filename:
                obj.box3d = os.path.join(dirpath, filename)
            elif "Persons" in filename:
                obj.person = os.path.join(dirpath, filename)
            elif "_gtF" in filename or "_gtC" in filename:
                obj.gt = os.path.join(dirpath, filename)

        toRemove = []
        for value in images.values():
            if not value.find(imagesFiles, suffix):
                toRemove.append(value.id)
        for key in toRemove:
            del images[key]
        #print(self.images)
        self.images= list(images.values())
        pass
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            return [self.__getitem__(v) for v in values]
        group = self.images[index]

        img = torchvision.io.read_image(group.img, torchvision.io.ImageReadMode.RGB).float()/255.0
        #r = img[0].clone()
        #b=img[2].clone()
        #img[0],img[2]=r,b
        det = Detection()
        if group.gt is not None:
            f = open(group.gt)
            data = json.load(f)
            f.close()
            objects:List = data["objects"]
            for obj in objects:
                label = obj["label"]
                id = CitiscapesDetection.getId(label)
                polygon = np.array(obj["polygon"])

                x1 = polygon[:,0].min() 
                x2 = polygon[:,0].max() 
                y1 = polygon[:,1].min() 
                y2 = polygon[:,1].max() 
                box = Box2d()
                box.x =x1
                box.y =y1
                box.w =x2-x1
                box.h =y2-y1
                box.c = id
                box.cn = label
                if not CitiscapesDetection.isBanned(id):
                    det.boxes2d.append(box)
        if group.gt is None and group.person is not None:
            f = open(group.gt)
            data = json.load(f)
            f.close()
            objects:List = data["objects"]
            for obj in objects:
                label = obj["label"]
                bbox2d = obj["bbox"]
                box = Box2d()
                box.x =bbox2d[0]
                box.y =bbox2d[1]
                box.w =bbox2d[2]
                box.h =bbox2d[3]
                box.c = id
                box.cn = label
                if not CitiscapesDetection.isBanned(id):
                    det.boxes2d.append(box)

        if group.box3d is not None:
            f = open(group.box3d)
            data = json.load(f)
            f.close()
            objects:List = data["objects"]
            for obj in objects:
                label = obj["label"]
                id = CitiscapesDetection.getId(label)
                bbox2d = obj["2d"]["modal"]
                box = Box2d()
                box.x =bbox2d[0]
                box.y =bbox2d[1]
                box.w =bbox2d[2]
                box.h =bbox2d[3]
                box.c = id
                box.cn = label
                if not CitiscapesDetection.isBanned(id):
                    det.boxes2d.append(box)
                _3d = obj["3d"]

                center = _3d["center"]
                rotation = _3d["rotation"]
                size = _3d["dimensions"]

                from scipy.spatial.transform import Rotation
                q = np.array(rotation)
                rotation = Rotation.from_quat(q)
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

                box = Box3d()
                box.c = id
                box.cn = label
                box.x = minx
                box.y = miny
                box.z = minz

                box.w = maxx-minx
                box.h = maxy-miny
                box.d = maxz-minz
                if not CitiscapesDetection.isBanned(id):
                    det.boxes3d.append(box)

        # for b in ann:
        #     box = Box2d()
        #     box.x = b["bbox"][0]
        #     box.y = b["bbox"][1]
        #     box.w = b["bbox"][2]
        #     box.h = b["bbox"][3]
        #     box.c = b["category_id"]
        #     det.boxes2d.append(box)
        citiSamp = Sample()
        citiSamp.setImage(img)
        citiSamp.setTarget(det)

        return citiSamp

