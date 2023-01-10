from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision
from ..datasets.Sample import Sample

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
    def scale(self,x=1.0,y=1.0):
        newBox = Box2d()
        newBox.x = self.x*x
        newBox.y = self.y*y
        newBox.w = self.w*x
        newBox.h = self.h*y
        newBox.c = self.c
        newBox.cn = self.cn
        newBox.cf = self.cf
        return newBox



    def __str__(self) -> str:
        return f"Box2d[x:{self.x},y:{self.y},w:{self.w},h:{self.h},class:{self.c},confidence:{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


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
    def scale(self,x=1.0,y=1.0, z=1.0):
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
        return newBox

    def __init__(self) -> None:
        self.x = self.y = self.w = self.h = self.z = self.d = 0
        self.c = 0
        self.cf = 0
        self.cn = ""

    def __str__(self) -> str:
        return f"Box3d[c:{self.x},y:{self.y},z:{self.z},w:{self.w},h:{self.h},d:{self.d},class:{self.c},confidence{self.cf}]"

    def __repr__(self) -> str:
        return self.__str__()


class Detection:
    boxes2d: List[Box2d]
    boxes3d: List[Box3d]

    def __init__(self) -> None:
        self.boxes2d = []
        self.boxes3d = []
        
    def scale(self,x=1.0,y=1.0):
        newDet = Detection()
        newDet.boxes2d = [b.scale(x,y) for b in self.boxes2d]
        newDet.boxes3d = list([b.scale() for b in self.boxes3d])
        return newDet

    def fromTorchVision(torchVisionResult, dataset=None):
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

    def filter(self, th):
        newVal = Detection()
        newVal.boxes2d = [x for x in self.boxes2d if x.cf > th]
        newVal.boxes3d = [x for x in self.boxes3d if x.cf > th]
        return newVal
    def c(self,c):
        d = Detection()
        d.boxes2d = [x for x in self.boxes2d if int(x.c) == int(c)]
        d.boxes3d = [x for x in self.boxes3d if int(x.c) == int(c)]
        return d
    def onImage(self, sample: Sample, colors:List[Tuple[int,int,int]]=None):
        if isinstance(sample,Sample):
            img = (sample.getRGB()*255.0).byte()
        elif isinstance(sample,torch.Tensor):
            img = sample
        else :
            raise Exception("Argument sample must be sample or tensor")
        target = self.toTorchVisionTarget("cpu")
        if len(self.boxes2d) > 0:
            labels = [b.cn for b in self.boxes2d]
            if colors is not None:
                colors = [c for c in colors]
                i=0
                while len(colors) < len(labels):
                    colors.append(colors[i])
                    i+=1
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=4, colors=colors)
                
                pass
            else:
                img = torchvision.utils.draw_bounding_boxes(img,target["boxes"],labels, width=4)
        return img
    def toX1Y1X2Y2C(self,device="cpu"):
        ret= torch.tensor([[x.x,x.y,x.x+x.w,x.y+x.h,x.c-1]for x in self.boxes2d]).to(device)
        if(len(ret.shape) == 1):
            ret = ret.view(0,5)
        return ret
    def toX1Y1X2Y2CFC(self,device="cpu"):
        ret= torch.tensor([[x.x,x.y,x.x+x.w,x.y+x.h,x.cf,x.c-1]for x in self.boxes2d]).to(device)
        if(len(ret.shape) == 1):
            ret = ret.view(0,6)
        return ret
    def toTorchVisionTarget(self, device="cpu"):
        boxes = []
        labels = []
        scores = []
        for box in self.boxes2d:
            if box.w<1:
                box.w=2
            if box.h<1:
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

    def NMS(self, overlapThresh = 0.4) :
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

        return newDetection
    def NMS_Pytorch(self,thresh_iou : float=0.4):
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

        return newDetection

