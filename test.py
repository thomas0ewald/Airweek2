import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import numpy as np
import time
from cv_bridge import CvBridge
import cv2


import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


data_array_camera = np.empty((0, 2), float)
data_array_scan = np.empty((0, 2), float)

def build_pose(image):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DENSEPOSE_DEMO = True  # Aktiviere den Dense Pose-Modus

    predictor = DefaultPredictor(cfg)

    outputs = predictor(image)
    # Zugriff auf die erkannten dichten Posen
    densepose_outputs = outputs["instances"].to("cpu")

    v = Visualizer(image[:, :, ::-1], scale=1.2)
    out = v.draw_instance_predictions(densepose_outputs)
    cv2.imshow("Dense Pose Estimation", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(densepose_outputs)
    # cv2.imshow("Dense Pose Estimation", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)




def callback_camera(msg):
    bridge = CvBridge()

    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    build_pose(cv_image)
    cv2.imwrite('oupuuuut.jpg', cv_image)

    global data_array_camera
    data_array_camera = np.append(data_array_camera, [[time.time(), msg.data]], axis=0)

def callback_scan(msg):
    global data_array_scan
    data_array_scan = np.append(data_array_scan, [[time.time(), msg]], axis=0)

def save_and_shutdown():
    print('im here')
    # rclpy.shutdown()
    global data_array_camera
    global data_array_scan

    np.save('camera_data.npy', data_array_camera)
    np.save('scan_data.npy', data_array_scan)
    print('ferig!!')

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

    timer = node.create_timer(300, lambda: save_and_shutdown())
    
    rclpy.spin(node)

    node.destroy_node()

if __name__ == '__main__':
    main()
