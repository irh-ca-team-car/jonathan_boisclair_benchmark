data = "/mnt/DATA/records/fourth_try.db3"
interested_topics = ["/soul/roof/lidar/points","/flir/raw","/zed_wrapper_fl/left/image_rect_color"]

from interface.datasets.RosbagDataset import RosbagDataset
import cv2
dataset = RosbagDataset(data, interested_topics, flir_topic="/flir/raw", rgb_topic="/zed_wrapper_fl/left/image_rect_color")

sample = dataset[0]
print(len(dataset))

for sample in dataset:
    sample.show(sample.getRGB(),True, name="RBG")
    sample.show(sample.getThermal(),True, name="Thermal")

sample.getLidar().view()