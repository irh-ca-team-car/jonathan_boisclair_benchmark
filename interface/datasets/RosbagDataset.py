from typing import List, Literal, Union
from . import Sample, LidarSample
import torch
import sqlite3
import zlib
import zstd
import rclpy.serialization
from cv_bridge import CvBridge
from ..adapters.OpenCV import CVAdapter
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
class RosbagGroup:
        def __init__(self) -> None:
            self.id_img = None
            self.id_flir = None
            self.id_pcl = None

class RosbagDataset:
    
    data : List[RosbagGroup]
    def __init__(self, rosbagfile, topics, flir_topic=None, rgb_topic=None) -> None:
     
        self.rosbagfile = rosbagfile
        self.topics = topics
        self.data = []
        self.flir_topic = flir_topic
        self.rgb_topic = rgb_topic
        self.pcl_topic = None

        self.parse()

    def maybePush(self, group: RosbagGroup) -> RosbagGroup:
        valid = True
        if self.flir_topic is not None:
            if group.id_flir is None:
                valid = False
        if self.rgb_topic is not None:
            if group.id_img is None:
                valid = False
        if self.pcl_topic is not None:
            if group.id_pcl is None:
                valid = False
        if valid:
            self.data.append(group)
            return RosbagGroup()
        return group
    def parse(self):  
        self.types=dict()
        self._zlib=dict()
        self._zstd=dict()
        con = sqlite3.connect(
            'file:'+self.rosbagfile+'?mode=ro', uri=True)
        cur = con.cursor()
        for row in cur.execute('SELECT id,name,type,serialization_format FROM topics'):
            try:
                if row[1] in self.topics:
                    namespace=row[2].split("/")
                    module = __import__(namespace[0])
                    if(not hasattr(module,"msg")):
                        module = __import__(namespace[0]+".msg")
                    type = getattr(module.msg,namespace[2])
                    self.types[row[0]]=type
                    self._zlib[row[0]] = "zlib" in row[3]
                    self._zstd[row[0]] = "zstd" in row[3]

                    if row[1] == self.flir_topic:
                        self.flir_topic = row[0]
                    if row[1] == self.rgb_topic:
                        self.rgb_topic = row[0]
                    if self.rgb_topic is None and self.flir_topic is None and type.__name__ == "Image":
                        self.rgb_topic = row[0]

                    if type.__name__ == "PointCloud2":
                        self.pcl_topic = row[0]

                    print(row[1], row[2])
            except BaseException as e:
                print("Could not playback ",row[1],"due to type not found:[",row[2],"]", e)
        con.close()

        con = sqlite3.connect(
            'file:'+self.rosbagfile+'?mode=ro', uri=True)
        cur = con.cursor()
        try:
            s = RosbagGroup()
            for row in cur.execute('SELECT topic_id,timestamp,data,id FROM messages'):
                if row[0] in self.types:
                    type = self.types[row[0]] 
                    if type.__name__ == "Image":
                        if row[0] == self.flir_topic:
                            s.id_flir= row[3]
                        elif row[0] == self.rgb_topic:
                            s.id_img= row[3]
                        else:
                            s.id_img= row[3]
                    elif  type.__name__ == "PointCloud2":
                        s.id_pcl= row[3]
                    s=self.maybePush(s)
        except BaseException as e:
            print("Playback failure",e)
        con.close()
        return
    def get(self,group: RosbagGroup) -> Sample:
        samp = Sample()
        con = sqlite3.connect(
            'file:'+self.rosbagfile+'?mode=ro', uri=True)
        cur = con.cursor()

        if group.id_img is not None:
            for row in cur.execute('SELECT topic_id,timestamp,data,id FROM messages where id = '+str(group.id_img)):
                data = row[2]
                if self._zlib[row[0]]:
                    data = zlib.decompress(data)
                if self._zstd[row[0]]:
                    data = zstd.decompress(data)
                data = rclpy.serialization.deserialize_message(data, Image)
                samp.setImage(CVAdapter().toPytorch(CvBridge().imgmsg_to_cv2(data,"bgr8")))
        if group.id_flir is not None:
            for row in cur.execute('SELECT topic_id,timestamp,data,id FROM messages where id = '+str(group.id_flir)):
                data = row[2]
                if self._zlib[row[0]]:
                    data = zlib.decompress(data)
                if self._zstd[row[0]]:
                    data = zstd.decompress(data)
                data = rclpy.serialization.deserialize_message(data, Image)
                samp.setThermal(CVAdapter().toPytorch(CvBridge().imgmsg_to_cv2(data,"mono16")))
        if group.id_pcl is not None:
            for row in cur.execute('SELECT topic_id,timestamp,data,id FROM messages where id = '+str(group.id_pcl)):
                data = row[2]
                if self._zlib[row[0]]:
                    data = zlib.decompress(data)
                if self._zstd[row[0]]:
                    data = zstd.decompress(data)
                data = rclpy.serialization.deserialize_message(data, PointCloud2)
                lidar:PointCloud2 = data
                fields = [f.name for f in lidar.fields]
                pts = pc2.read_points_list(data, skip_nans=True)

                x = torch.tensor([p.x for p in pts]).view(-1,1)
                y = torch.tensor([p.y for p in pts]).view(-1,1)
                z = torch.tensor([p.z for p in pts]).view(-1,1)
                i = torch.tensor([p.intensity for p in pts]).view(-1,1) if "intensity" in fields else None
                            
                r = None
                g = None
                b = None
                if "rgb" in fields:
                    print(torch.tensor([p.rgb for p in pts]).view(-1,1))

                samp._lidar = LidarSample.fromXYZIRingRGBAT(x,y,z,i, None,r,g,b,None,None)
        con.close()
        return samp
        
    def __len__(self):
        return len(self.data)

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
            return self.get(self.data[index])

