import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import time
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import datetime
from rclpy.time import Time

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


import pandas as pd

from Regression import Regression

data_array_camera = np.empty((0, 2), float)
data_array_scan = np.empty((0, 2), float)
last_array_scan = np.empty((0, 2), float)

#globals for file
data_keypoints_for_file = []
data_distances_for_file = []
data_intensities_for_file = []

class Gateway:

    def __init__(self):
        rclpy.init()
        
        self.node = rclpy.create_node('my_subscriber')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.camera_subscription = self.node.create_subscription(
            Image,
            '/theta_z1/rgb',
            self.callback_camera,
            10
        )

        
        self.scan_subscription = self.node.create_subscription(
            LaserScan,
            '/ouster/scan',
            self.callback_scan,
            qos,
        )

        self.publisher = self.node.create_publisher(
            Image,
            "nie_ohne_mein_team",
            10
        )

        self.keypoints_publisher = self.node.create_publisher(
            MarkerArray,
            "nie_ohne_mein_team_keypoints",
            10
        )

        self.bridge = CvBridge()

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

        self.keypoints_dataa = []


        timer = self.node.create_timer(120, lambda: self.write_file())
        
        rclpy.spin(self.node)
        self.node.destroy_node()

    def get_keypoints(self, keypoints):
        # keypoints.header.frame_id = "base_link"
        # keypoints.header.stamp = Time.now()
    # Füge Felder für x, y und z Koordinaten hinzu
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        z = keypoints[:, 2]

        pc_msg = PointCloud2()
        pc_msg.header = Header()
        # pc_msg.header.stamp = time.now()
        pc_msg.header.frame_id = 'grid'

        pc_msg.height = 1
        pc_msg.width = len(x)
        # fields = [
        #     ('x', 0, PointField.FLOAT32, 1),
        #     ('y', 4, PointField.FLOAT32, 1),
        #     ('z', 8, PointField.FLOAT32, 1)
        # ]
        # pc_msg.fields = [PointField(*field) for field in fields]

        pc_msg.is_bigendian = False
        pc_msg.point_step = 12  # Jede Punktinformation ist 12 Bytes groß (3 * 4 Bytes)
        pc_msg.row_step = pc_msg.point_step * len(x)
        pc_msg.is_dense = True

        # Füge die Punkte zur Nachricht hinzu
        points = np.column_stack((x, y, z))
        pc_msg.data = np.asarray(points, np.float32).tostring()

        self.keypoints_publisher.publish(pc_msg)


    def numpy_array_to_markerarray(self, numpy_array2):
        marker_array = MarkerArray()

        # np.append(numpy_array2, [0,0,0])
        # scores = instances.scores.cpu().numpy()

        marker = Marker()
        marker.header.frame_id = "base_link"  # Rahmen des Koordinatensystems
        marker.header.stamp = self.node.get_clock().now().to_msg() 
        marker.id = 10000000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.scale.x = 0.05  # Durchmesser des Sphärenmarkers
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = float(127)
        marker.color.g = float(0)
        marker.color.b = float(255)

        marker_array.markers.append(marker)

        count = 0
        colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

        # Erstelle Marker für jeden Punkt in der Punktwolke
        for idx, numpy_array in enumerate(numpy_array2):
            for point in numpy_array:
                marker = Marker()
                marker.header.frame_id = "base_link"  # Rahmen des Koordinatensystems
                marker.header.stamp = self.node.get_clock().now().to_msg() 
                marker.id = count
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = (float(point[2]) / 50)
                marker.pose.position.y = float(point[0]) / 50
                marker.pose.position.z = (-float(point[1]) / 50) + 3
                marker.scale.x = 0.05  # Durchmesser des Sphärenmarkers
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                marker.color.r = float(colours[idx][0])
                marker.color.g = float(colours[idx][1])
                marker.color.b = float(colours[idx][2])

                marker_array.markers.append(marker)
                count += 1

            
        self.keypoints_publisher.publish(marker_array)


    def build_pose(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"]
        densepose_outputs = outputs["instances"].to("cpu")

        v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                
        densepose_outputs.remove("pred_classes")  # Entferne die vorhergesagten Klassen (für die Anzeige ohne Klassenbeschriftungen)
        densepose_outputs.remove("scores")

        out = v.draw_instance_predictions(densepose_outputs)

        keypoints_arr = densepose_outputs.pred_keypoints.numpy()
        # print(keypoints_arr)a_boxes.tensor.cpu().numpy()
        # keypoints = instances.pred_keypoints[0].cpu().numpy()
        # scores = instances.scores.cpu().numpy()
        # print(keypoints)
        self.keypoints_dataa = keypoints_arr

        # for box in boxes:
        #     x_min, y_min, x_max, y_max = box
        #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        self.publish_image(out.get_image()[:, :, ::-1])
        # self.publish_image(image)

        # self.get_keypoints(keypoints)
        self.numpy_array_to_markerarray(keypoints_arr)



    def publish_image(self, image):
        converted_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.publisher.publish(converted_img)
        print("image published")


    def callback_camera(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.build_pose(cv_image)
        
        # global data_array_camera
        # data_array_camera = np.append(data_array_camera, [[time.time(), msg.data]], axis=0)

    def callback_scan(self, msg):
        # global data_array_scan
        # data_array_scan = np.append(data_array_scan, [[time.time(), msg]], axis=0)
        ranges = np.array(msg.ranges)
        intensities = np.array(msg.intensities)
        # print(len(last_array_scan))
        print(f"Intensity: {intensities}")
        print(f"Distance: {ranges}")
        print(f"Keypoints: {self.keypoints_dataa}")

        # prepare file
        global data_keypoints_for_file
        global data_distances_for_file
        global data_intensities_for_file

        data_keypoints_for_file.append(self.keypoints_dataa)
        data_distances_for_file.append(ranges)
        data_intensities_for_file.append(intensities)


    def save_and_shutdown(self):
        print('im here')
        # rclpy.shutdown()
        global data_array_camera
        global data_array_scan

        np.save('camera_data.npy', data_array_camera)
        np.save('scan_data.npy', data_array_scan)
        print('ferig!!')

    def write_file(self):
        # prepare file
        global data_keypoints_for_file
        global data_distances_for_file
        global data_intensities_for_file

        data = {
        'Keypoint': data_keypoints_for_file,
        'Distance': data_distances_for_file,
        'Intensity': data_intensities_for_file
        }
        df = pd.DataFrame(data)

        # Pfad festlegen, wo die Datei gespeichert werden soll
        file_path = 'data.csv'

        # DataFrame als CSV-Datei speichern
        df.to_csv(file_path, index=False)

        print("successfull")

gateway = Gateway()
