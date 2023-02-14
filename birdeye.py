from interface.datasets.detection.KittiMultiview import KittiMultiviewDetection
import fiftyone.zoo as foz
import fiftyone as fo
import torch
data = "/mnt/DATA/records/fourth_try.db3"
interested_topics = ["/soul/roof/lidar/points","/flir/raw","/zed_wrapper_fl/left/image_rect_color"]

from interface.datasets.RosbagDataset import RosbagDataset
import cv2
#dataset = RosbagDataset(data, interested_topics, flir_topic="/flir/raw", rgb_topic="/zed_wrapper_fl/left/image_rect_color")
dataset = KittiMultiviewDetection("train", dataset_dir="/media/boiscljo/LinuxData/Datasets/km")

for sample in dataset:
    xyzrgb=sample.getLidar().XYZRGB()

    birdseyeSample = sample.getLidar().birdseye(sample, scale=10)

    image = birdseyeSample.detection.onImage(birdseyeSample, colors=[(255,0,0)])
    sample.show(image,True)

exit(0)
x = xyzrgb[:,0]
y = xyzrgb[:,1]
z = xyzrgb[:,2]

if torch.nonzero(xyzrgb[:,3:6]).shape[0] ==0:
    xyzrgb[:,3:6]=1

xmin = x.min()
xmax = x.max()

print(xmin,xmax)

ymin = y.min()
ymax = y.max()

print(ymin,ymax)

zmin = z.min()
zmax = z.max()

print(zmin,zmax)
import math
scale = 25

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

x = (x *scale).long() + -cx
y = (y *scale).long() + -cy

cwh[:,x,y] = xyzrgb[:,3:6].permute(1,0).float()

sample.show(cwh,True,"bird")