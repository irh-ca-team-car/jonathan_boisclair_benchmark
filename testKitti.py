from interface.datasets.detection.KittiMultiview import KittiMultiviewDetection
import fiftyone.zoo as foz
import fiftyone as fo

dataset = KittiMultiviewDetection("train", dataset_dir="/media/boiscljo/LinuxData/Datasets/km")

sample = dataset[0]
print(sample)
print(sample.detection)
print(sample.getLidar().view())