import cv2
from geometry_msgs.msg import Twist
import datetime
from rclpy.node import Node 
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 
from drive_BOT import Car
import rclpy
from std_msgs.msg import String  # Import message type for traffic sign

# YOLOv9 and DeepSort imports
import torch
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLOv9Webcam:
    def __init__(self, weights_path, device=''):
        self.device = select_device(device)
        self.model = attempt_load(weights_path)
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img, img_size=640):
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img):
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        return pred

    def draw_boxes(self, pred, img, tracks):
        for track in tracks:
            if track.is_confirmed() and track.time_since_update < 1:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                label = f'{self.names[class_id]} {track_id}'

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img

class CombinedProcessor(Node):
    def __init__(self):
        super().__init__('combined_processor')
        
        # Publishers and Subscribers
        self.publisher_img_read = self.create_publisher(Image, '/imgread', 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.subscriber_img = self.create_subscription(Image, 'video_frames', self.process_data, 10)
        self.subscriber_traffic_sign = self.create_subscription(String, 'traffic_sign', self.traffic_sign_callback, 10)
        self.color_sub = self.create_subscription(Image, 'video_frames', self.color_callback, 10)

        # Timer for sending commands
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        # Initialize necessary components
        self.bridge = CvBridge()
        self.Car = Car()
        self.p_err = 0
        self.vel = ''
        self.velocity = Twist()

        # YOLO and DeepSort for object detection and tracking
        self.yolo = YOLOv9Webcam('/home/datnguyen/ros2_ws/Cuonwlam/ROS2-autonomous-driving-computer-vision/self_driving/Thesis2.pt')
        self.tracker = DeepSort(max_age=30)

        self.color_frame = None

    def traffic_sign_callback(self, msg):
        self.vel = msg.data
        
    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)
        
    def process_data(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        self.p_err, Angle, Speed, img = self.Car.drive_car(frame, self.p_err, self.vel)

        self.velocity.angular.z = Angle
        self.velocity.linear.x = Speed

        cv2.imshow("Frame", img)
        cv2.waitKey(1)

        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.publisher_img_read.publish(img_msg)
        
    def color_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frames()

    def process_frames(self):
        if self.color_frame is not None:
            preprocessed_img = self.yolo.preprocess(self.color_frame)
            pred = self.yolo.detect(preprocessed_img)

            detect = []
            detected_labels = []  # Initialize detected labels list
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(preprocessed_img.shape[2:], det[:, :4], self.color_frame.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        detect.append([[x1, y1, x2-x1, y2-y1], conf.item(), int(cls.item())])

            tracks = self.tracker.update_tracks(detect, frame=self.color_frame)
            img = self.yolo.draw_boxes(pred, self.color_frame, tracks)
            cv2.imshow('Processed Image', img)
            cv2.waitKey(1)

            # Debug: Print the detected labels
            print(f"Detected labels: {detected_labels}")

            # Update tracks
            tracks = self.tracker.update_tracks(detect, frame=self.color_frame)

            # Draw boxes and tracks
            img = self.yolo.draw_boxes(pred, self.color_frame, tracks)

            # Check for specific labels to execute tasks
            for label in detected_labels:
                print(f"Checking label: {label}")  # Debug: Print each label being checked
                if label == "no right turn for cars":
                    print("Stop sign detected! Executing stop task.")  # Debug: Print action taken
                    # You can call a specific method or function here to perform actions like stopping the car.
                    # e.g., self.execute_stop()

def main(args=None):
    rclpy.init(args=args)
    node = CombinedProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
