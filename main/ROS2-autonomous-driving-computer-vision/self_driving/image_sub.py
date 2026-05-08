import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLOv9Webcam:
    def __init__(self, weights_path, device=''):
        self.device = select_device(device)
        self.model = attempt_load(weights_path)  # Load model with specified path
        self.model.to(self.device)  # Move model to the desired device
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img, img_size=320):
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert from BGR to RGB, format 3xHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # Convert from uint8 to float16/32
        img /= 255.0  # Normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        return img

    def detect(self, img):
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)  # Apply NMS
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

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.color_sub = self.create_subscription(Image, 'video_frames', self.color_callback, 10)
        self.yolo = YOLOv9Webcam('/home/datnguyen/yolov9/Thesis2.pt')
        self.tracker = DeepSort(max_age=30)
        self.color_frame = None

    def color_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frames()

    def process_frames(self):
        if self.color_frame is not None:
            preprocessed_img = self.yolo.preprocess(self.color_frame)
            pred = self.yolo.detect(preprocessed_img)
            detect = []
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(preprocessed_img.shape[2:], det[:, :4], self.color_frame.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        detect.append([[x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())])
            tracks = self.tracker.update_tracks(detect, frame=self.color_frame)
            img = self.yolo.draw_boxes(pred, self.color_frame, tracks)
            cv2.imshow('Processed Image', img)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

