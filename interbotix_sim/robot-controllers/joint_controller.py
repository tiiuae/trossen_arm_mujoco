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
Joint Space Controller for Interbotix Arms.

This module provides a controller class that consumes joint space actions
and commands the robot arm accordingly.
"""

from typing import List, Optional
import argparse

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class JointController:
    """
    Controller for commanding robot arm joints directly.

    This class provides methods to consume joint position actions and
    command the robot arm in joint space.
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
        Initialize the Joint Controller.

        Args:
            robot_model: Interbotix Arm model (e.g., 'wx200', 'wx250s')
            group_name: Joint group name for the arm (default: 'arm')
            gripper_name: Name of the gripper joint (default: 'gripper')
            robot_name: Custom robot name for multi-robot setups
            moving_time: Time in seconds for joint movements (default: 2.0)
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
        print(f"Joint Controller started for {self.robot_model}")

    def shutdown(self):
        """Shutdown the robot system."""
        robot_shutdown()
        print("Joint Controller shutdown")

    def execute_action(
        self,
        joint_positions: List[float],
        gripper_position: Optional[float] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Execute a joint space action.

        Args:
            joint_positions: List of joint positions in radians
            gripper_position: Optional gripper position (0.0=closed, 1.0=open)
            moving_time: Optional override for movement time
            accel_time: Optional override for acceleration time
            blocking: Whether to wait for movement completion

        Returns:
            bool: True if action executed successfully, False otherwise
        """
        # Validate joint positions
        if len(joint_positions) != self.num_joints:
            print(
                f"Error: Expected {self.num_joints} joint positions, "
                f"got {len(joint_positions)}"
            )
            return False

        # Command the arm joints
        success = self.bot.arm.set_joint_positions(
            joint_positions=joint_positions,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )

        if not success:
            print("Warning: Joint positions may be out of limits")
            return False

        # Command the gripper if specified
        if gripper_position is not None and self.use_gripper:
            self.set_gripper(gripper_position, blocking=blocking)

        return True

    def execute_trajectory(
        self,
        trajectory: List[List[float]],
        gripper_trajectory: Optional[List[float]] = None,
        moving_time: Optional[float] = None,
        accel_time: Optional[float] = None,
    ) -> bool:
        """
        Execute a trajectory of joint positions.

        Args:
            trajectory: List of joint position lists
            gripper_trajectory: Optional list of gripper positions
            moving_time: Optional override for movement time per waypoint
            accel_time: Optional override for acceleration time

        Returns:
            bool: True if trajectory executed successfully, False otherwise
        """
        if not trajectory:
            print("Error: Empty trajectory")
            return False

        for i, joint_positions in enumerate(trajectory):
            gripper_pos = None
            if gripper_trajectory is not None and i < len(gripper_trajectory):
                gripper_pos = gripper_trajectory[i]

            success = self.execute_action(
                joint_positions=joint_positions,
                gripper_position=gripper_pos,
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=True,
            )

            if not success:
                print(f"Error: Failed to execute waypoint {i}")
                return False

        return True

    def get_joint_positions(self) -> List[float]:
        """
        Get current joint positions.

        Returns:
            List[float]: Current joint positions in radians
        """
        return self.bot.arm.get_joint_positions()

    def get_joint_velocities(self) -> List[float]:
        """
        Get current joint velocities.

        Returns:
            List[float]: Current joint velocities in rad/s
        """
        return self.bot.arm.get_joint_velocities()

    def get_joint_efforts(self) -> List[float]:
        """
        Get current joint efforts.

        Returns:
            List[float]: Current joint efforts (PWM or current)
        """
        return self.bot.arm.get_joint_efforts()

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
        # Assuming gripper opens from 0 to some max value
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
    """Example usage of the Joint Controller."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Joint Space Controller for Interbotix Arms'
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
    controller = JointController(
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

        # Execute a single action
        print("Executing joint action...")
        joint_action = [-0.274969220161438, -0.9670658111572266, 0.8647124767303467, -0.005227804183959961, 0.2915276288986206, -0.3617055416107178]
        controller.execute_action(joint_action, gripper_position=1.0)

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
