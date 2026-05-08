import cv2
from geometry_msgs.msg import Twist
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
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
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.subscriber_img = self.create_subscription(CompressedImage, 'video_frames', self.process_data, 10)
        self.subscriber_traffic_sign = self.create_subscription(String, 'traffic_sign', self.traffic_sign_callback, 10)
        self.color_sub = self.create_subscription(CompressedImage, 'video_frames', self.color_callback, 10)

        # Timer for sending commands
        cmd_timer_period = 0.1  # Adjust the time period as needed
        self.cmd_timer = self.create_timer(cmd_timer_period, self.send_cmd_vel)

        # Timer for processing image and traffic sign data
        process_timer_period = 0.1  # Adjust the time period as needed
        self.process_timer = self.create_timer(process_timer_period, self.process_sensors_data)

        # Initialize necessary components
        self.bridge = CvBridge()
        self.Car = Car()
        self.p_err = 0
        self.vel = ''
        self.velocity = Twist()

        self.yolo = YOLOv9Webcam('/home/datnguyen/ros2_ws/Cuonwlam/ROS2-autonomous-driving-computer-vision/self_driving/Hailon.pt')
        self.tracker = DeepSort(max_age=30)

        self.color_frame = None
        self.can_publish_cmd_vel = True
        self.red_light_detected = False 

        # Cờ để theo dõi khi nhãn "no parking on even days" đã được phát hiện và xử lý
        self.no_parking_detected = False  # Cờ này sẽ kiểm tra xem hàm đã được gọi chưa

    def traffic_sign_callback(self, msg):
        self.vel = msg.data
        
    def send_cmd_vel(self):
        # Kiểm tra nếu cờ can_publish_cmd_vel là True thì mới thực hiện publish
        if self.can_publish_cmd_vel:
            self.publisher.publish(self.velocity)
        else:
            print("Publish tạm thời bị tắt.")

    def enable_cmd_vel_publish(self):
        self.can_publish_cmd_vel = True

    def disable_cmd_vel_publish(self):
        self.can_publish_cmd_vel = False

    def process_sensors_data(self):
        # Process both image and traffic sign data
        # Add any necessary logic for this combined processing
        pass

    def process_data(self, data):
        # Decode compressed image
        np_arr = np.frombuffer(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        self.p_err, Angle, Speed, img = self.Car.drive_car(frame, self.p_err, self.vel)

        self.velocity.angular.z = Angle
        self.velocity.linear.x = Speed
        
        width = 700
        height = 700
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Tạo cửa sổ có thể thay đổi kích thước
        cv2.resizeWindow("Frame", width, height)
        cv2.imshow("Frame", img)
        cv2.waitKey(1)

    def color_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.color_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_frames()

    def stop_car_for_red_light(self):
        if self.cmd_timer is not None:
            self.cmd_timer.cancel()  # Tạm dừng timer cho việc gửi lệnh vận tốc
            self.cmd_timer = None

        self.disable_cmd_vel_publish
        self.red_light_detected = True
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)
        print("Stop moving due to red light.")

    def green_light(self):
        if self.cmd_timer is not None:
            self.cmd_timer.cancel()  # Tạm dừng timer cho việc gửi lệnh vận tốc
            self.cmd_timer = None

        self.disable_cmd_vel_publish()
        self.red_light_detected = False
        self.velocity.linear.x = 0.15
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)
        print("Tăng tốc lên 0.2 và duy trì trong 10 giây")
        self.accel_timer = self.create_timer(2.0, self.reset_speed)

    def reset_speed(self):
        if self.accel_timer is not None:
            self.accel_timer.cancel()
            self.accel_timer = None
        
        if not self.red_light_detected:
            self.publisher.publish(self.velocity)
            print("Đặt lại tốc độ.")

    def cross_bridge(self):
        self.disable_cmd_vel_publish()
        self.velocity.linear.x = 0.15
        self.velocity.angular.z = -0.035
        self.publisher.publish(self.velocity)
        print("Tăng tốc lên 0.2 và duy trì trong 3 giây")
        if not hasattr(self, 'bridge_timer') or self.bridge_timer is None:
            self.bridge_timer = self.create_timer(3.0, self.reset_speed_2)

    def reset_speed_2(self):
        if self.bridge_timer is not None:
            self.bridge_timer.cancel()
            self.bridge_timer = None
        self.enable_cmd_vel_publish()

    def no_left_turn_action(self):
        if self.cmd_timer is not None:
            self.cmd_timer.cancel()  # Tạm dừng timer cho việc gửi lệnh vận tốc
            self.cmd_timer = None

        self.disable_cmd_vel_publish
        self.red_light_detected = True
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)

        if not self.no_parking_detected:  # Kiểm tra cờ trước khi gọi hàm
        #    if self.cmd_timer is not None:
        #        self.cmd_timer.cancel()
        #        self.cmd_timer = None
        

            self.velocity.linear.x = 0.0
            self.velocity.angular.z = 0.0
            self.publisher.publish(self.velocity)
            self.disable_cmd_vel_publish()
        print("Robot dừng lại.")
            
        if not hasattr(self, 'turn_timer') or self.turn_timer is None:
            self.turn_timer = self.create_timer(3.0, self.go_straight_after_turn)

            # Sau khi gọi hành động, đặt cờ thành True để tránh gọi lại hàm
            
        self.no_parking_detected = True

    def go_straight_after_turn(self):
        if self.turn_timer is not None:
            self.turn_timer.cancel()
            self.turn_timer = None  # Reset turn_timer
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = -0.5
        self.publisher.publish(self.velocity)
        print("Tăng tốc lên 0.1 và quay phải nhẹ")
        if not hasattr(self, 'speed_reset_timer') or self.speed_reset_timer is None:
            self.speed_reset_timer = self.create_timer(3.3, self.reset_speed_1)

    def reset_speed_1(self):
        if self.speed_reset_timer is not None:
            self.speed_reset_timer.cancel()
            self.speed_reset_timer = None  # Reset speed_reset_timer
        self.velocity.linear.x = 0.1
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)
        if not hasattr(self, 'stop_timer') or self.stop_timer is None:
            self.stop_timer = self.create_timer(3.5, self.turn_left)

    def turn_left(self):
        if self.turn_timer is not None:
            self.turn_timer.cancel()
            self.turn_timer = None  # Reset turn_timer
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.5
        self.publisher.publish(self.velocity)
        print("Xe quay trái.")
        if not hasattr(self, 'speed_reset_timer') or self.speed_reset_timer is None:
            self.speed_reset_timer = self.create_timer(3.3, self.stop_car)

    def stop_car(self):
        if self.turn_timer is not None:
            self.turn_timer.cancel()
            self.turn_timer = None
        if self.speed_reset_timer is not None:
            self.speed_reset_timer.cancel()
            self.speed_reset_timer = None
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)
        print("Xe đã dừng hoàn toàn.")

    def turn_around(self):
        if self.cmd_timer is not None:
            self.cmd_timer.cancel()  # Tạm dừng timer cho việc gửi lệnh vận tốc
            self.cmd_timer = None

        self.disable_cmd_vel_publish
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 1.5
        self.publisher.publish(self.velocity)
        print("Turning around.")
        if not hasattr(self, 'turn_around_timer') or self.turn_around_timer is None:
            self.turn_around_timer = self.create_timer(2.5, self.reset_speed_3)

    def reset_speed_3(self):
        if self.turn_around_timer is not None:
            self.turn_around_timer.cancel()
            self.turn_around_timer = None
        self.enable_cmd_vel_publish()

    def crosswalk(self):
        self.disable_cmd_vel_publish()
        self.velocity.linear.x = 0.05
        self.velocity.angular.z = 0.0
        self.publisher.publish(self.velocity)
        print("Crosswalk detected.")
        self.crosswalk_timer = self.create_timer(2.0, self.reset_speed_4)
    
    def reset_speed_4(self):
        if self.crosswalk_timer is not None:
            self.crosswalk_timer.cancel()
            self.crosswalk_timer = None
        self.enable_cmd_vel_publish()


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

                        # Get the label text from class index
                        label_text = self.yolo.names[int(cls.item())]
                        detected_labels.append(label_text)  # Add detected label to list
                                    
            # Debug: Print the detected labels
            print(f"Detected labels: {detected_labels}")

            # Update tracks
            tracks = self.tracker.update_tracks(detect, frame=self.color_frame)

            # Draw boxes and tracks
            img = self.yolo.draw_boxes(pred, self.color_frame, tracks)
            cv2.imshow('Processed Image', img)
            cv2.waitKey(1)

            # Check for specific labels to execute tasks
            for label in detected_labels:
                print(f"Checking label: {label}")  # Debug: Print each label being checked
                if label == "red light":
                    print("Red light detected! Stopping car.")  # Debug: Print action taken
                    self.stop_car_for_red_light()  # Gọi phương thức để dừng xe
                elif label == "green light":
                    print("Green light detected! Accelerating car.")  # Debug: Print action taken
                    self.green_light()
                elif label == "no parking on even days" and not self.no_parking_detected:
                    print("Stopping car and turning right.")  # Debug: Print action taken
                    self.no_left_turn_action()
                elif label == "bridge":
                    print("Crossing the bridge")  # Debug: Print action taken
                    self.cross_bridge() 
                elif label == "no right turn for cars":
                    print("Turning the car around")  # Debug: Print action taken
                    self.turn_around() 
                elif label == "walking":
                    print("Detected the walking")  # Debug: Print action taken
                    self.crosswalk()

def main(args=None):
    rclpy.init(args=args)
    node = CombinedProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
