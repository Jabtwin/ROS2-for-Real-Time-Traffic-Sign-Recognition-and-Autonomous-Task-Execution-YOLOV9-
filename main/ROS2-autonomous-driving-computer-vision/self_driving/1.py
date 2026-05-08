import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLOv9RealSense:
    def __init__(self, weights_path, device='cpu'):
        self.device = select_device(device)
        self.model = attempt_load(weights_path)
        self.model.to(self.device)  # Move the model to the desired device
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img, img_size=640):
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # (0 - 255) to (0.0 - 1.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img):
        pred = self.model(img)[0]
        pred = non_max_suppression(pred)
        return pred

    def draw_boxes(self, pred, img):
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape, det[:, :4], img.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        return img

# Initialize Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize YOLOv9 with RealSense
yolo = YOLOv9RealSense('/home/datnguyen/yolov9/best.pt')

try:
    while True:
        # Get frame from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Preprocess image
        img = yolo.preprocess(frame)

        # Object detection
        pred = yolo.detect(img)

        # Draw bounding boxes
        frame = yolo.draw_boxes(pred, frame)

        # Display the image
        cv2.imshow('RealSense', frame)
        cv2.waitKey(1)

finally:
    pipeline.stop()
