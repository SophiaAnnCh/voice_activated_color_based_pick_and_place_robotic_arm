#!/usr/bin/env python3
"""
Voice command node with push-to-record button.
Press ENTER to start recording, releases when done.
Only records when robot is not moving.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import threading
from panda_nlp.voice_nlp_core import VoiceControlledRoboticsNLP


class VoiceCommandNode(Node):

    def __init__(self):
        super().__init__('voice_command_node')

        self.publisher_ = self.create_publisher(String, '/voice_command', 10)

        # Listen to robot busy state published by pick_and_place
        self.robot_busy = False
        self.busy_sub = self.create_subscription(
            Bool, '/robot_busy', self.busy_callback, 10
        )

        self.get_logger().info("Loading NLP pipeline...")
        self.system = VoiceControlledRoboticsNLP(duration=5)
        self.get_logger().info("=" * 50)
        self.get_logger().info("Voice Command Node Ready")
        self.get_logger().info("Press ENTER to record a 5-second voice command")
        self.get_logger().info("Recording is disabled while robot is moving")
        self.get_logger().info("=" * 50)

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def busy_callback(self, msg):
        self.robot_busy = msg.data
        if self.robot_busy:
            self.get_logger().info("Robot is moving — recording disabled")
        else:
            self.get_logger().info("Robot is idle — press ENTER to record")

    def _input_loop(self):
        while rclpy.ok():
            try:
                input("\n>>> Press ENTER to record a voice command...\n")

                if self.robot_busy:
                    self.get_logger().warn("Robot is currently moving. Wait until it finishes.")
                    continue

                self._record_and_publish()

            except EOFError:
                break
            except KeyboardInterrupt:
                break

    def _record_and_publish(self):
        self.get_logger().info("Recording for 5 seconds — speak now!")
        try:
            text, _, sdc = self.system.process_voice_command()

            if not text.strip():
                self.get_logger().warn("Nothing heard. Try again.")
                return

            if sdc.color is None:
                self.get_logger().warn(
                    f"No color detected in: '{text}'\n"
                    "Say something like: 'pick red sphere' or 'grab blue ball'"
                )
                return

            sdc_json = json.dumps(sdc.to_dict())
            self.publisher_.publish(String(data=sdc_json))
            self.get_logger().info(f"Command understood: color={sdc.color}, text='{text}'")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
