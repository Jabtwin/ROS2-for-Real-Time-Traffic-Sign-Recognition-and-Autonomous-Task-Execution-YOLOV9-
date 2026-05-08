# yolov9.py
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLOv9:
    def __init__(self, weights_path, device='cpu'):
        self.device = select_device(device)
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = np.random.randint(0, 255, (len(self.names), 3))

        self.detection_threshold = 0.7

    def preprocess(self, img, img_size=640):
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, rearrange color channels
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img):
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5)
        return pred

    def draw_boxes(self, bgr_frame, detections):
        for det in detections:
            if len(det):
                det[:, :4] = scale_boxes(bgr_frame.shape[2:], det[:, :4], bgr_frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    if conf > self.detection_threshold:
                        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), self.colors[int(cls)], 2)
                        cv2.putText(bgr_frame, f'{self.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[int(cls)], 2)
        return bgr_frame
