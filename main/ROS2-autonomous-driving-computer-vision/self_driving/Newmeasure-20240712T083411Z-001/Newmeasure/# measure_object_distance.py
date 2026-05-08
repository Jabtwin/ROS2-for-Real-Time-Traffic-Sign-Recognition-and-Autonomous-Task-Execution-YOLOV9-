# measure_object_distance.py
import cv2
from realsense_camera import RealsenseCamera
from yolov9 import YOLOv9

# Load Realsense camera and YOLOv9 model
rs = RealsenseCamera()
yolo = YOLOv9(weights_path="/home/datnguyen/yolov9/best.pt")

while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    if not ret:
        continue

    # Preprocess and detect using YOLOv9
    preprocessed_img = yolo.preprocess(bgr_frame)
    detections = yolo.detect(preprocessed_img)

    # Draw bounding boxes on the frame
    bgr_frame = yolo.draw_boxes(bgr_frame, detections)

    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

rs.release()
cv2.destroyAllWindows()
