import sqlite3
import os
import rclpy
import rclpy.serialization
from cv_bridge import CvBridge
import zlib
import zstd
from interface.datasets import Sample, LidarSample
from interface.adapters.OpenCV import CVAdapter
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import torch
data = "/mnt/DATA/records/fourth_try.db3"
interested_topics = ["/soul/roof/lidar/points","/zed_wrapper_fl/left/image_rect_color"]

types = dict()
_zlib = dict()
_zstd = dict()
rosbagfile = data
if(rosbagfile is None or len(rosbagfile) == 0):
    raise Exception("Rosbag parameter is empty")
if not os.path.exists(rosbagfile):
    raise Exception("Rosbag parameter does not exists")
con = sqlite3.connect('file:'+rosbagfile+'?mode=ro', uri=True)
cur = con.cursor()

for row in cur.execute('SELECT id,name,type,serialization_format FROM topics'):
            #print(row[0], row[1],row[2])
            try:
                if row[1] in interested_topics:
                    namespace=row[2].split("/")
                    module = __import__(namespace[0])
                    if(not hasattr(module,"msg")):
                        module = __import__(namespace[0]+".msg")
                    type = getattr(module.msg,namespace[2])
                    types[row[0]]=type
                    _zlib[row[0]] = "zlib" in row[3]
                    _zstd[row[0]] = "zstd" in row[3]
                    print(row[1], row[2])
            except:
                print("Could not playback ",row[1],"due to type not found:[",row[2],"]")
con.close()

def data():      
        con = sqlite3.connect(
            'file:'+rosbagfile+'?mode=ro', uri=True)
        cur = con.cursor()
        try:
            s = Sample()
            print("Starting from beginning")
            for row in cur.execute('SELECT topic_id,timestamp,data,id FROM messages'):
                if row[0] in types:
                    type = types[row[0]] 
                    data = row[2]
                    if _zlib[row[0]]:
                        data = zlib.decompress(data)
                    if _zstd[row[0]]:
                        data = zstd.decompress(data)
                    data = rclpy.serialization.deserialize_message(data, type)
                    if data.__class__.__name__ == "Image":
                        s.setImage(CVAdapter().toPytorch(CvBridge().imgmsg_to_cv2(data,"bgr8")))
                    elif  data.__class__.__name__ == "PointCloud2":
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

                        s._lidar = LidarSample.fromXYZIRingRGBAT(x,y,z,i, None,r,g,b,None,None)

                 
            print("Completed playback")
        except BaseException as e:
            print("Playback failure",e)
        con.close()
        return

data()