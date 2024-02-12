#general imports
import datetime
import pandas as pd
import torch
import numpy as np
import time

#import ros2 modules
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

#import detectron2 - model for people detection
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from cv_bridge import CvBridge
import cv2

from Regression import Regression



data_array_camera = np.empty((0, 2), float)
data_array_scan = np.empty((0, 2), float)
last_array_scan = np.empty((0, 2), float)

#globals for file
data_keypoints_for_file = []
data_distances_for_file = []
data_intensities_for_file = []

class Gateway:
    """Gateway class to connect to sensors"""
    def __init__(self):
        rclpy.init()
        
        #create node
        self.node = rclpy.create_node('my_subscriber')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        #create subscriber for camera data
        self.camera_subscription = self.node.create_subscription(
            Image,
            '/theta_z1/rgb',
            self.callback_camera,
            10
        )
        
        #create subscriber for lidar data
        self.scan_subscription = self.node.create_subscription(
            LaserScan,
            '/ouster/scan',
            self.callback_scan,
            qos,
        )

        #create publisher for visualisation of Image in rviz2
        self.publisher = self.node.create_publisher(
            Image,
            "nie_ohne_mein_team",
            10
        )

        #create publisher for visualisation of MarkerArray in rviz2
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

        self._keypoints_data = []


        # collect data for 120 seconds and save data to file
        timer = self.node.create_timer(120, lambda: self.write_file())
        
        rclpy.spin(self.node)
        self.node.destroy_node()

    def numpy_array_to_markerarray(self, numpy_array2):
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "base_link"  # Name - in rviz2 change frame_id in globals to base_link
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

        # Create new entry for every Point
        for idx, numpy_array in enumerate(numpy_array2):
            for point in numpy_array:
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.node.get_clock().now().to_msg() 
                marker.id = count
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = (float(point[2]) / 50) #scale datapoints to display on grid
                marker.pose.position.y = float(point[0]) / 50
                marker.pose.position.z = (-float(point[1]) / 50) + 3
                marker.scale.x = 0.05
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
                
        densepose_outputs.remove("pred_classes")  #remove labels of predicted classes for display purposes
        densepose_outputs.remove("scores") #remove confidence score for display

        #draw predictions on image
        out = v.draw_instance_predictions(densepose_outputs)

        keypoints_arr = densepose_outputs.pred_keypoints.numpy()
        self._keypoints_data = keypoints_arr
        #publish image
        self.publish_image(out.get_image()[:, :, ::-1])
        #create markerarray for 3d grid in rviz2
        self.numpy_array_to_markerarray(keypoints_arr)


    def publish_image(self, image):
        converted_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.publisher.publish(converted_img)
        print("image published")


    def callback_camera(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.build_pose(cv_image)
        

    def callback_scan(self, msg):
        ranges = np.array(msg.ranges)
        intensities = np.array(msg.intensities)
        print(f"Intensity: {intensities}") #how strong the object reflects - can be used to filter unwanted objects
        print(f"Distance: {ranges}") #distance to camera
        print(f"Keypoints: {self._keypoints_data}") #skeleton points of detected people

        # prepare file
        global data_keypoints_for_file
        global data_distances_for_file
        global data_intensities_for_file

        data_keypoints_for_file.append(self._keypoints_data)
        data_distances_for_file.append(ranges)
        data_intensities_for_file.append(intensities)


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

        file_path = 'data.csv'
        #save df to .csv
        df.to_csv(file_path, index=False)

        print("successfully saved to file")
        
gateway = Gateway()
