import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

class YOLOv9RealSense:
    def __init__(self, weights_path, device=''):
        self.device = select_device(device)
        self.model = attempt_load(weights_path)
        self.model.to(self.device)  # Move model to the device
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()

        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                                 max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg_deep.DEEPSORT.MAX_AGE,
                                 n_init=cfg_deep.DEEPSORT.N_INIT,
                                 nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                                 use_cuda=self.half)

    def preprocess(self, img, img_size=640):
        img_resized = cv2.resize(img, (img_size, img_size))
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)
        img_resized = torch.from_numpy(img_resized).to(self.device)
        img_resized = img_resized.half() if self.half else img_resized.float()  # uint8 to fp16/32
        img_resized /= 255.0  # (0 - 255) to (0.0 - 1.0)
        if img_resized.ndimension() == 3:
            img_resized = img_resized.unsqueeze(0)
        return img_resized

    def detect(self, img):
        pred = self.model(img)[0]
        pred = non_max_suppression(pred)
        return pred

    def draw_boxes(self, pred, ori_img):
        xywh_bboxs = []
        confs = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(ori_img.shape, det[:, :4], ori_img.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    xywh_bbox = (xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item())
                    xywh_bboxs.append(xywh_bbox)
                    confs.append([conf.item()])

                    cv2.rectangle(ori_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(ori_img, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        
        # Debug: Print the shapes of inputs to update method
        print(f'xywhs shape: {xywhs.shape}, confss shape: {confss.shape}, ori_img shape: {ori_img.shape}')

        outputs = self.deepsort.update(xywhs, confss, ori_img)  # Pass the original image

        for j, (output) in enumerate(outputs):
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_w = output[2] - output[0]
            bbox_h = output[3] - output[1]
            track_id = output[4]
            cv2.rectangle(ori_img, (bbox_left, bbox_top), (bbox_left + bbox_w, bbox_top + bbox_h), (0, 255, 0), 2)
            cv2.putText(ori_img, f'ID {track_id}', (bbox_left, bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        return ori_img

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
        img_resized = yolo.preprocess(frame)

        # Object detection
        pred = yolo.detect(img_resized)

        # Draw bounding boxes and track objects
        frame = yolo.draw_boxes(pred, frame)  # Pass the original frame here

        # Display the image
        cv2.imshow('RealSense', frame)
        cv2.waitKey(1)

finally:
    pipeline.stop()
