from interface.datasets import Sample, Detection, Box2d
import torchvision
detGT = Detection()
box1 = Box2d()
box1.x = 0.5
box1.y = 0.5
box1.w = 0.1
box1.h = 0.1
detGT.boxes2d.append(box1)
box1 = Box2d()
box1.x = 0.7
box1.y = 0.5
box1.w = 0.1
box1.h = 0.1
detGT.boxes2d.append(box1)
box1 = Box2d()
box1.x = 0.2
box1.y = 0.5
box1.w = 0.1
box1.h = 0.1
detGT.boxes2d.append(box1)


det2 = Detection()
box1 = Box2d()
box1.x = 0.5
box1.y = 0.5
box1.w = 0.1
box1.h = 0.05
box2 = Box2d()
box2.x = 0.5
box2.y = 0.5
box2.w = 0.06
box2.h = 0.05
det2.boxes2d.append(box1)
det2.boxes2d.append(box2)
box1 = Box2d()
box1.x = 0.2
box1.y = 0.5
box1.w = 0.1
box1.h = 0.1
det2.boxes2d.append(box1)

d3 = det2.BoxIOU(detGT)
print(d3.shape)

mask = d3.max(1).values > 0.5
d4=d3[mask]

print(d4,d4.max(1).indices)
print(det2.Box2dMask(mask))
print(detGT.Box2dIndices(d4.max(1).indices))

s = Sample.Example()
print(s.detection.boxes2d[0].EvaluateDistance(s))