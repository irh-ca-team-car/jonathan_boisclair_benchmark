from visualization_msgs.msg import Marker

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from random import randint
class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('adapter'+str(randint(10000,100000)), allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.pub_soul = self.create_publisher(Marker,"/marker/soul",1)
        self.pub_campus = self.create_publisher(Marker,"/marker/campus",1)
        self.timer = self.create_timer(1/30,self.send)
    def send(self):
        #self.sendCampus()
        self.sendSoul()
    def sendCampus(self):
        self.sendGeneric(self.pub_campus,"campus","package://campus_dae/map.dae")
    def sendSoul(self):
        self.sendGeneric(self.pub_soul,"car","package://kia_soul/soul.dae")
    def sendGeneric(self,pub,frame,mesh):
        m=Marker()
        m.type = Marker.MESH_RESOURCE
        m.mesh_resource = mesh
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        #if frame == "car":
        #    m.color.a =1.0
        #    m.color.r =1.0
        #    m.color.g =1.0
        #    m.color.b =1.0
        m.action = Marker.ADD
        m.scale.x=1.0
        m.scale.y=1.0
        if frame == "campus":
            m.scale.x = 1.05
            m.scale.y = 1.03
            m.pose.position.x =0.0
            m.pose.position.y =-28.0
        m.scale.z=1.0
        m.mesh_use_embedded_materials = True
        #m.mesh_use_embedded_materials = frame == "campus"
        #print(m)
        pub.publish(m)
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
