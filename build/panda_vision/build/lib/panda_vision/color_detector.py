#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Publisher
        self.coords_pub = self.create_publisher(String, '/color_coordinates', 10)

        # OpenCV bridge
        self.bridge = CvBridge()

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsic parameters (from your SDF)
        self.fx = 585.0
        self.fy = 588.0
        self.cx = 320.0
        self.cy = 160.0

        self.get_logger().info("Color Detector Node Started with TF2 lookup transform")

    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges (HSV)
        color_ranges = {
            "R": [(0, 120, 70), (10, 255, 255)],
            "G": [(55, 200, 200), (60, 255, 255)],
            "B": [(90, 200, 200), (128, 255, 255)]
        }

        for color_id, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Noise removal
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Only keep near-circular objects
                if circularity < 0.7:
                    continue

                # Valid sphere detected
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cx_pix, cy_pix = int(x), int(y)
                radius = int(radius)

                # Draw detected sphere
                cv2.circle(frame, (cx_pix, cy_pix), radius, (0, 255, 255), 2)
                cv2.circle(frame, (cx_pix, cy_pix), 4, (255, 255, 255), -1)
                cv2.putText(
                    frame,
                    color_id,
                    (cx_pix - 20, cy_pix - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Convert pixel -> camera frame
                Z = 0.1
                Y = (cx_pix - self.cx) * Z / self.fx * -10
                X = (cy_pix - self.cy) * Z / self.fy

                try:
                    t = self.tf_buffer.lookup_transform(
                        "panda_link0",
                        "camera_link",
                        rclpy.time.Time(),
                        timeout=Duration(seconds=1.0),
                    )

                    trans = np.array([
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                    ])

                    rot = [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ]

                    T = tf_transformations.quaternion_matrix(rot)
                    T[:3, 3] = trans

                    pt_cam = np.array([X, Y, Z, 1.0])
                    pt_base = T @ pt_cam

                    if color_id == "B":
                        pt_base[1] -= 0.0215
                    elif color_id == "G":
                        pt_base[1] += 0.01

                    msg_str = f"{color_id},{pt_base[0]:.3f},{pt_base[1]:.3f},{pt_base[2]:.3f}"
                    self.coords_pub.publish(String(data=msg_str))
                    self.get_logger().info(msg_str)

                except Exception as e:
                    self.get_logger().warn(f"TF transform failed: {e}")

                break  # Stop after first valid sphere

        # Show image in window
        try:
            cv2.namedWindow("Color Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Color Detection", 640, 320)
            cv2.imshow("Color Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"OpenCV display error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
