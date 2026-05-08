import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, 'video_frames', 10)
        self.bridge = CvBridge()
        self.capture = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.publish_image)  # Timer-based publishing at 10 Hz

    def publish_image(self):
        ret, img = self.capture.read()
        if ret:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.publisher.publish(img_msg)
            self.get_logger().info('Published image frame.')  # Debugging line to confirm publishing
        else:
            self.get_logger().error("Could not grab a frame!")

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
