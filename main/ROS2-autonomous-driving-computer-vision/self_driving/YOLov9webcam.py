import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLOv9Webcam:
    def __init__(self, weights_path, device=''):
        self.device = select_device(device)
        self.model = attempt_load(weights_path)  # Load model with specified path
        self.model.to(self.device)  # Move the model to the desired device
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img, img_size=640):
        img = cv2.resize(img, (img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch dimension
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

# Initialize YOLOv9 with specific weight path
yolo = YOLOv9Webcam('/home/datnguyen/ros2_ws/Cuonwlam/ROS2-autonomous-driving-computer-vision/self_driving/CUoicungdat.pt')

# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        preprocessed_img = yolo.preprocess(frame)
        pred = yolo.detect(preprocessed_img)

        detect = []
        detected_labels = []  # Initialize detected labels list
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(preprocessed_img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detect.append([[x1, y1, x2-x1, y2-y1], conf.item(), int(cls.item())])
                    
                    # Get the label text from class index
                    label_text = yolo.names[int(cls.item())]
                    detected_labels.append(label_text)  # Add detected label to list

        # Debug: Print the detected labels
        print(f"Detected labels: {detected_labels}")

        # Update tracks
        tracks = tracker.update_tracks(detect, frame=frame)

        # Draw boxes and tracks
        frame = yolo.draw_boxes(pred, frame, tracks)

        # Check for specific labels to execute tasks
        for label in detected_labels:
            print(f"Checking label: {label}")  # Debug: Print each label being checked
            if label == "no right turn for cars":
                print("Stop sign detected! Executing stop task.")  # Debug: Print action taken
                # You can call a specific method or function here to perform actions like stopping the car.
                # e.g., self.execute_stop()

        # Display the frame
        cv2.imshow('Webcam YOLOv9 DeepSort', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
