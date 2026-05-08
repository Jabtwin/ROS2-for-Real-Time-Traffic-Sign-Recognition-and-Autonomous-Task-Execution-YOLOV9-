import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np  # Import numpy to handle RealSense frame data
import cv2

class CamPublisher(Node):
    def __init__(self):
        super().__init__('cam_publisher')
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the pipeline to stream color frames
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        self.bridge = CvBridge()

    def timer_callback(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        # Convert RealSense frame to NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert OpenCV image to ROS Image message and publish
        msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing video frame')

def main(args=None):
    rclpy.init(args=args)
    cam_publisher = CamPublisher()
    rclpy.spin(cam_publisher)
    cam_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
