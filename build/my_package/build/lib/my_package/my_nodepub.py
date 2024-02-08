import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import numpy as np
import time

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


data_array_camera = np.array([])
data_array_scan = np.array([])

def callback_camera(msg):
    global data_array_camera
    data_array_camera = np.append(data_array_camera, [[time.time(), msg.data]], axis=0)

def callback_scan(msg):
    global data_array_scan
    data_array_scan = np.append(data_array_scan, [[time.time(), msg]], axis=0)

def save_and_shutdown():
    rclpy.shutdown()
    global data_array_camera
    global data_array_scan
    np.save('camera_data', data_array_camera)
    np.save('scan_data', data_array_scan)

def main():
    rclpy.init()

    node = rclpy.create_node('my_subscriber')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


    topic_name= '/theta_z1/rgb'

    msg_type = Image

    camera_subscription = node.create_subscription(
        Image,
        '/theta_z1/rgb',
        callback_camera,
        10
    )

    scan_subscription = node.create_subscription(
        LaserScan,
        '/ouster/scan',
        callback_scan,
        qos,
    )

    timer = node.create_timer(20, lambda: save_and_shutdown())

    node.destroy_node()

if __name__ == '__main__':
    main()
