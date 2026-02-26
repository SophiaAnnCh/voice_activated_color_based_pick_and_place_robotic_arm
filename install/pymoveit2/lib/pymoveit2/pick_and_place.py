#!/usr/bin/env python3
"""
Pick and place node - runs continuously, accepts multiple voice commands.
Publishes /robot_busy to block voice recording during motion.

Based on working prototype with added safety improvements:
- Slower, controlled movements to prevent knocking balls
- Cartesian paths for vertical descent/lift
- Pre-grasp safety position
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

import math


class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")

        self.target_color = None
        self.is_moving = False
        self.latest_coords = {}

        # Separate callback groups
        self.moveit_cb = ReentrantCallbackGroup()
        self.sub_cb = MutuallyExclusiveCallbackGroup()

        # Arm MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.moveit_cb,
        )
        
        # Slower speeds for smoother, safer motion
        self.moveit2.max_velocity = 0.1  # Same as working prototype
        self.moveit2.max_acceleration = 0.1  # Same as working prototype

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.moveit_cb,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        # Publishers / Subscribers
        self.busy_pub = self.create_publisher(Bool, '/robot_busy', 10)

        self.voice_sub = self.create_subscription(
            String, '/voice_command', self.voice_callback, 10,
            callback_group=self.sub_cb,
        )
        self.coords_sub = self.create_subscription(
            String, '/color_coordinates', self.coords_callback, 10,
            callback_group=self.sub_cb,
        )

        # Joint positions
        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        self.home_joints  = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0, math.radians(92.0), math.radians(50.0)]
        self.drop_joints  = [math.radians(-155.0), math.radians(30.0), math.radians(-20.0),
                             math.radians(-124.0), math.radians(44.0), math.radians(163.0), math.radians(7.0)]

        # Approach offset (same as working prototype)
        self.approach_offset = 0.31

        self.get_logger().info("Moving to start position...")
        self._set_busy(True)
        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()
        self._set_busy(False)
        self.get_logger().info("Ready! Waiting for voice commands...")

    def _set_busy(self, busy: bool):
        self.is_moving = busy
        self.busy_pub.publish(Bool(data=busy))

    def _try_execute(self):
        if self.is_moving:
            return
        if self.target_color is None:
            return
        if self.target_color not in self.latest_coords:
            self.get_logger().info(f"Target {self.target_color} set — waiting for coordinates...")
            return

        coords = self.latest_coords[self.target_color]
        color = self.target_color
        self.target_color = None
        self._set_busy(True)

        self.get_logger().info(
            f"Executing pick and place for {color} at "
            f"[{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]"
        )
        thread = Thread(target=self._execute_pick_and_place, args=(coords,), daemon=True)
        thread.start()

    def voice_callback(self, msg):
        import json
        self.get_logger().info(f"Voice callback fired: {msg.data}")

        if self.is_moving:
            self.get_logger().warn("Robot is moving — voice command ignored")
            return

        try:
            data = json.loads(msg.data)
            color = data.get("color")
            if not color:
                self.get_logger().warn("Voice command had no color — ignored")
                return

            self.target_color = color[0].upper()
            self.get_logger().info(f"Target color set to: {self.target_color}")
            self._try_execute()

        except Exception as e:
            self.get_logger().error(f"Voice callback error: {e}")

    def coords_callback(self, msg):
        try:
            color_id, x, y, z = msg.data.split(",")
            color_id = color_id.strip().upper()
            self.latest_coords[color_id] = [float(x), float(y), float(z)]

            if not self.is_moving:
                self._try_execute()

        except Exception as e:
            self.get_logger().error(f"Coords callback error: {e}")

    def _execute_pick_and_place(self, coords):
        try:
            # Use the EXACT same coordinate calculation as working prototype
            pick_position = [coords[0], coords[1], coords[2] - 0.60]
            quat_xyzw = [0.0, 1.0, 0.0, 0.0]
            
            # Calculate approach position (same as prototype)
            approach_position = [
                pick_position[0],
                pick_position[1],
                pick_position[2] - self.approach_offset
            ]

            self.get_logger().info("Step 1/11: Moving to home...")
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 2/11: Moving above target...")
            self.moveit2.move_to_pose(position=pick_position, quat_xyzw=quat_xyzw)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 3/11: Opening gripper...")
            self.gripper.open()
            self.gripper.wait_until_executed()

            self.get_logger().info("Step 4/11: Descending to object (Cartesian)...")
            # USE CARTESIAN PATH for straight-line descent to avoid hitting adjacent balls
            self.moveit2.move_to_pose(
                position=approach_position,
                quat_xyzw=quat_xyzw,
                cartesian=True  # KEY IMPROVEMENT: straight down, no arc
            )
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 5/11: Closing gripper...")
            self.gripper.close()
            self.gripper.wait_until_executed()

            self.get_logger().info("Step 6/11: Lifting back up (Cartesian)...")
            # USE CARTESIAN PATH for straight-line lift to avoid dragging into other balls
            self.moveit2.move_to_pose(
                position=pick_position, 
                quat_xyzw=quat_xyzw,
                cartesian=True  # KEY IMPROVEMENT: straight up, no arc
            )
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 7/11: Moving to home...")
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 8/11: Moving to drop position...")
            self.moveit2.move_to_configuration(self.drop_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("Step 9/11: Opening gripper...")
            self.gripper.open()
            self.gripper.wait_until_executed()

            self.get_logger().info("Step 10/11: Closing gripper...")
            self.gripper.close()
            self.gripper.wait_until_executed()

            self.get_logger().info("Step 11/11: Returning to start...")
            self.moveit2.move_to_configuration(self.start_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("=" * 50)
            self.get_logger().info("Sequence complete! Ready for next voice command.")
            self.get_logger().info("=" * 50)

        except Exception as e:
            self.get_logger().error(f"Pick and place failed: {e}")

        finally:
            self._set_busy(False)


def main():
    rclpy.init()
    node = PickAndPlace()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
