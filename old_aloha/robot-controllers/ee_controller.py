#!/usr/bin/env python3

# Copyright 2024 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
End-Effector Space Controller for Interbotix Arms.

This module provides a controller class that consumes end-effector space actions
and commands the robot arm accordingly using inverse kinematics.
"""

from typing import List, Optional, Union
import argparse
import numpy as np

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class EEController:
    """
    Controller for commanding robot arm end-effector.

    This class provides methods to consume end-effector pose actions and
    command the robot arm in task space using inverse kinematics.
    """

    def __init__(
        self,
        robot_model: str,
        group_name: str = 'arm',
        gripper_name: str = 'gripper',
        robot_name: Optional[str] = None,
        moving_time: float = 2.0,
        accel_time: float = 0.3,
        gripper_pressure: float = 0.5,
        use_gripper: bool = True,
    ):
        """
        Initialize the End-Effector Controller.

        Args:
            robot_model: Interbotix Arm model (e.g., 'wx200', 'wx250s')
            group_name: Joint group name for the arm (default: 'arm')
            gripper_name: Name of the gripper joint (default: 'gripper')
            robot_name: Custom robot name for multi-robot setups
            moving_time: Time in seconds for movements (default: 2.0)
            accel_time: Time in seconds for acceleration/deceleration (default: 0.3)
            gripper_pressure: Gripper pressure fraction 0-1 (default: 0.5)
            use_gripper: Whether to initialize gripper control (default: True)
        """
        self.robot_model = robot_model
        self.group_name = group_name
        self.use_gripper = use_gripper

        # Initialize the manipulator
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            group_name=group_name,
            gripper_name=gripper_name if use_gripper else None,
            robot_name=robot_name,
            moving_time=moving_time,
            accel_time=accel_time,
            gripper_pressure=gripper_pressure,
        )

        # Get number of joints in the arm group
        self.num_joints = self.bot.arm.get_number_of_joints()

    def start(self):
        """Start the robot system."""
        robot_startup()
        print(f"EE Controller started for {self.robot_model}")

    def shutdown(self):
        """Shutdown the robot system."""
        robot_shutdown()
        print("EE Controller shutdown")

    def execute_action_components(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: Optional[float] = None,
        gripper_position: Optional[float] = None,
        custom_guess: Optional[List[float]] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Execute an end-effector action using pose components.

        Args:
            x: X position in meters
            y: Y position in meters
            z: Z position in meters
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Optional yaw angle in radians
            gripper_position: Optional gripper position (0.0=closed, 1.0=open)
            custom_guess: Optional initial guess for IK solver
            moving_time: Optional override for movement time
            accel_time: Optional override for acceleration time
            blocking: Whether to wait for movement completion

        Returns:
            bool: True if action executed successfully, False otherwise
        """
        # Command the end-effector pose
        joint_solution, success = self.bot.arm.set_ee_pose_components(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            custom_guess=custom_guess,
            execute=True,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )

        if not success:
            print("Warning: IK solution may not be valid or pose unreachable")
            return False

        # Command the gripper if specified
        if gripper_position is not None and self.use_gripper:
            self.set_gripper(gripper_position, blocking=blocking)

        return True

    def execute_action_matrix(
        self,
        transformation_matrix: np.ndarray,
        gripper_position: Optional[float] = None,
        custom_guess: Optional[List[float]] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Execute an end-effector action using a transformation matrix.

        Args:
            transformation_matrix: 4x4 homogeneous transformation matrix
            gripper_position: Optional gripper position (0.0=closed, 1.0=open)
            custom_guess: Optional initial guess for IK solver
            moving_time: Optional override for movement time
            accel_time: Optional override for acceleration time
            blocking: Whether to wait for movement completion

        Returns:
            bool: True if action executed successfully, False otherwise
        """
        # Validate transformation matrix
        if transformation_matrix.shape != (4, 4):
            print(f"Error: Expected 4x4 matrix, got {transformation_matrix.shape}")
            return False

        # Command the end-effector pose
        joint_solution, success = self.bot.arm.set_ee_pose_matrix(
            T_sd=transformation_matrix,
            custom_guess=custom_guess,
            execute=True,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )

        if not success:
            print("Warning: IK solution may not be valid or pose unreachable")
            return False

        # Command the gripper if specified
        if gripper_position is not None and self.use_gripper:
            self.set_gripper(gripper_position, blocking=blocking)

        return True

    def execute_action(
        self,
        pose: Union[List[float], np.ndarray],
        gripper_position: Optional[float] = None,
        custom_guess: Optional[List[float]] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Execute an end-effector action (auto-detects format).

        Args:
            pose: Either [x, y, z, roll, pitch, yaw] or 4x4 transformation matrix
            gripper_position: Optional gripper position (0.0=closed, 1.0=open)
            custom_guess: Optional initial guess for IK solver
            moving_time: Optional override for movement time
            accel_time: Optional override for acceleration time
            blocking: Whether to wait for movement completion

        Returns:
            bool: True if action executed successfully, False otherwise
        """
        pose_array = np.array(pose)

        # Check if it's a transformation matrix
        if pose_array.shape == (4, 4):
            return self.execute_action_matrix(
                transformation_matrix=pose_array,
                gripper_position=gripper_position,
                custom_guess=custom_guess,
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=blocking,
            )
        # Check if it's pose components
        elif len(pose_array) == 6:
            x, y, z, roll, pitch, yaw = pose_array
            return self.execute_action_components(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                gripper_position=gripper_position,
                custom_guess=custom_guess,
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=blocking,
            )
        else:
            print(
                f"Error: Expected pose with 6 components or 4x4 matrix, "
                f"got shape {pose_array.shape}"
            )
            return False

    def execute_cartesian_trajectory(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        moving_time: Optional[float] = None,
        wp_moving_time: float = 0.2,
        wp_accel_time: float = 0.1,
        wp_period: float = 0.05,
    ) -> bool:
        """
        Execute a Cartesian trajectory (straight line in task space).

        Args:
            x: X displacement in meters
            y: Y displacement in meters
            z: Z displacement in meters
            roll: Roll displacement in radians
            pitch: Pitch displacement in radians
            yaw: Yaw displacement in radians
            moving_time: Total time for trajectory
            wp_moving_time: Time per waypoint
            wp_accel_time: Acceleration time per waypoint
            wp_period: Period between waypoints

        Returns:
            bool: True if trajectory executed successfully, False otherwise
        """
        success = self.bot.arm.set_ee_cartesian_trajectory(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            moving_time=moving_time,
            wp_moving_time=wp_moving_time,
            wp_accel_time=wp_accel_time,
            wp_period=wp_period,
        )

        if not success:
            print("Warning: Cartesian trajectory may not be fully reachable")
            return False

        return True

    def execute_trajectory(
        self,
        trajectory: List[Union[List[float], np.ndarray]],
        gripper_trajectory: Optional[List[float]] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
    ) -> bool:
        """
        Execute a trajectory of end-effector poses.

        Args:
            trajectory: List of poses (each can be [x,y,z,r,p,y] or 4x4 matrix)
            gripper_trajectory: Optional list of gripper positions
            moving_time: Optional override for movement time per waypoint
            accel_time: Optional override for acceleration time

        Returns:
            bool: True if trajectory executed successfully, False otherwise
        """
        if not trajectory:
            print("Error: Empty trajectory")
            return False

        for i, pose in enumerate(trajectory):
            gripper_pos = None
            if gripper_trajectory is not None and i < len(gripper_trajectory):
                gripper_pos = gripper_trajectory[i]

            success = self.execute_action(
                pose=pose,
                gripper_position=gripper_pos,
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=True,
            )

            if not success:
                print(f"Error: Failed to execute waypoint {i}")
                return False

        return True

    def get_ee_pose(self) -> np.ndarray:
        """
        Get current end-effector pose.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        return self.bot.arm.get_ee_pose()

    def get_ee_pose_command(self) -> np.ndarray:
        """
        Get commanded end-effector pose.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        return self.bot.arm.get_ee_pose_command()

    def get_joint_positions(self) -> List[float]:
        """
        Get current joint positions.

        Returns:
            List[float]: Current joint positions in radians
        """
        return self.bot.arm.get_joint_positions()

    def set_gripper(self, position: float, blocking: bool = True):
        """
        Set gripper position.

        Args:
            position: Gripper position (0.0=closed, 1.0=open)
            blocking: Whether to wait for movement completion
        """
        if not self.use_gripper:
            print("Warning: Gripper not initialized")
            return

        # Convert normalized position to gripper command
        if position >= 0.5:
            self.bot.gripper.release(delay=0.0 if not blocking else 1.0)
        else:
            self.bot.gripper.grasp(delay=0.0 if not blocking else 1.0)

    def go_to_home_pose(
        self,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ):
        """Move arm to home pose."""
        self.bot.arm.go_to_home_pose(
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )

    def go_to_sleep_pose(
        self,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ):
        """Move arm to sleep pose."""
        self.bot.arm.go_to_sleep_pose(
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )


def main():
    """Example usage of the EE Controller."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='End-Effector Space Controller for Interbotix Arms'
    )
    parser.add_argument(
        '--robot-model',
        type=str,
        default='vx300s',
        help='Interbotix robot model (default: vx300s)'
    )
    parser.add_argument(
        '--moving-time',
        type=float,
        default=1.5,
        help='Movement time in seconds (default: 1.5)'
    )
    parser.add_argument(
        '--accel-time',
        type=float,
        default=0.3,
        help='Acceleration time in seconds (default: 0.3)'
    )
    args = parser.parse_args()

    # Initialize controller
    controller = EEController(
        robot_model=args.robot_model,
        moving_time=args.moving_time,
        accel_time=args.accel_time,
    )

    # Start the robot
    controller.start()

    try:
        # Go to home pose
        print("Moving to home pose...")
        controller.go_to_home_pose()

        # Execute a single action using components
        print("Executing EE action (components)...")
        controller.execute_action_components(
            x=0.3,
            y=0.1,
            z=0.2,
            roll=0.0,
            pitch=0.5,
            yaw=0.0,
            gripper_position=1.0,
        )

        # Execute using pose vector
        print("Executing EE action (vector)...")
        pose = [0.25, -0.1, 0.25, 0.0, 0.8, 0.0]
        controller.execute_action(pose, gripper_position=0.0)

        # Execute a Cartesian trajectory
        print("Executing Cartesian trajectory...")
        controller.execute_cartesian_trajectory(
            x=0.05,
            y=0.05,
            z=-0.05,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            moving_time=2.0,
        )

        # Return to home and sleep
        print("Returning to home pose...")
        controller.go_to_home_pose()
        print("Moving to sleep pose...")
        controller.go_to_sleep_pose()

    finally:
        # Shutdown
        controller.shutdown()


if __name__ == '__main__':
    main()
