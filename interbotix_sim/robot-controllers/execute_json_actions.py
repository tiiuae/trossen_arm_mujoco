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
Execute actions from JSON files on a single VX300S arm.

Simplified script for single-arm control with left/right selection.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class SingleArmExecutor:
    """Simple executor for single VX300S arm."""

    def __init__(
        self,
        arm_side: str,
        moving_time: float = 2.0,
        accel_time: float = 0.3,
    ):
        """
        Initialize the Single Arm Executor.

        Args:
            arm_side: 'left' or 'right'
            moving_time: Time per action in seconds
            accel_time: Acceleration time in seconds
        """
        self.arm_side = arm_side
        self.moving_time = moving_time
        self.accel_time = accel_time

        print(f"Initializing VX300S {arm_side} arm")
        self.bot = InterbotixManipulatorXS(
            robot_model='vx300s',
            gripper_name='gripper',
            moving_time=moving_time,
            accel_time=accel_time,
        )

    def load_json_actions(self, json_path: str) -> Tuple[np.ndarray, dict]:
        """Load actions from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        actions = np.array(data['actions'], dtype=np.float32)
        metadata = {
            'shape': data.get('shape', actions.shape),
            'dtype': data.get('dtype', 'float32'),
        }

        print(f"Loaded {len(actions)} actions from {json_path}")
        print(f"Action shape: {actions.shape}")

        return actions, metadata

    def extract_arm_action(self, action: np.ndarray) -> Tuple[list, float]:
        """
        Extract joint positions and gripper for selected arm.

        JSON Format (14 values):
        [right(0-5), right_gripper(6), left(7-12), left_gripper(13)]

        Args:
            action: Full 14-element action array

        Returns:
            Tuple of (joint_positions, gripper_value)
        """
        action_flat = action.flatten()

        if self.arm_side == 'right':
            joints = action_flat[0:6].tolist()
            gripper = float(action_flat[6])
        else:  # left
            joints = action_flat[7:13].tolist()
            gripper = float(action_flat[13])

        return joints, gripper

    def extract_ee_action(self, action: np.ndarray) -> Tuple[list, float]:
        """
        Extract EE pose and gripper for selected arm.

        JSON Format (14 values for EE):
        [right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper,
         left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper]

        Args:
            action: Full 14-element action array

        Returns:
            Tuple of (ee_pose [x,y,z,roll,pitch,yaw], gripper_value)
        """
        action_flat = action.flatten()

        if self.arm_side == 'right':
            ee_pose = action_flat[0:6].tolist()
            gripper = float(action_flat[6])
        else:  # left
            ee_pose = action_flat[7:13].tolist()
            gripper = float(action_flat[13])

        return ee_pose, gripper

    def execute_action(self, action: np.ndarray, mode: str = 'joint') -> bool:
        """Execute a single action in joint or EE mode."""
        if mode == 'joint':
            return self.execute_joint_action(action)
        elif mode == 'ee':
            return self.execute_ee_action(action)
        else:
            print(f"Error: Unknown mode '{mode}'")
            return False

    def execute_joint_action(self, action: np.ndarray) -> bool:
        """Execute a joint space action."""
        joints, gripper = self.extract_arm_action(action)

        # Send joint command
        success = self.bot.arm.set_joint_positions(
            joint_positions=joints,
            blocking=True,
        )

        if not success:
            print("    ❌ Joint command failed")
            self.report_joint_limits(joints)
            return False

        # Control gripper
        if gripper < -0.5:
            self.bot.gripper.grasp(delay=0.0)
        else:
            self.bot.gripper.release(delay=0.0)

        return True

    def execute_ee_action(self, action: np.ndarray) -> bool:
        """Execute an end-effector space action."""
        ee_pose, gripper = self.extract_ee_action(action)

        # Extract pose components
        x, y, z = ee_pose[0:3]
        roll, pitch, yaw = ee_pose[3:6]

        # Send EE command
        _, success = self.bot.arm.set_ee_pose_components(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            execute=True,
            blocking=True,
        )

        if not success:
            print(f"    ❌ EE pose unreachable: [{x:.3f}, {y:.3f}, {z:.3f}] "
                  f"[{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
            return False

        # Control gripper
        if gripper < -0.5:
            self.bot.gripper.grasp(delay=0.0)
        else:
            self.bot.gripper.release(delay=0.0)

        return True

    def report_joint_limits(self, joints: list):
        """Report which joints are out of bounds."""
        group_info = self.bot.arm.group_info
        violations = []

        for idx, (joint_name, position) in enumerate(zip(group_info.joint_names, joints)):
            joint_index = self.bot.arm.info_index_map[joint_name]
            lower = group_info.joint_lower_limits[joint_index]
            upper = group_info.joint_upper_limits[joint_index]

            if position < lower or position > upper:
                violations.append(f"{joint_name}: {position:.3f} rad ({np.rad2deg(position):.1f}°) "
                                  f"[limit: {lower:.3f} to {upper:.3f}]")

        if violations:
            print("    Joint limit violations:")
            for v in violations:
                print(f"      • {v}")
        else:
            print("    All joints within limits (trajectory timing may be too fast)")

    def execute_trajectory(
        self,
        actions: np.ndarray,
        mode: str = 'joint',
        start_idx: int = 0,
        max_actions: int = None,
        delay: float = 0.0,
    ) -> int:
        """Execute a trajectory of actions."""
        total_actions = len(actions)
        end_idx = min(start_idx + max_actions, total_actions) if max_actions else total_actions
        num_actions = end_idx - start_idx

        print(f"\n{'='*60}")
        print(f"Executing {num_actions} actions [{start_idx} → {end_idx-1}]")
        print(f"Mode: {mode.upper()} | Arm: {self.arm_side.upper()} | "
              f"Timing: {self.moving_time}s move, {self.accel_time}s accel")
        print(f"{'='*60}\n")

        executed = 0
        failed = 0

        for i in range(start_idx, end_idx):
            action = actions[i]

            # Show progress inline
            progress_pct = ((i - start_idx + 1) / num_actions) * 100
            print(f"\rAction {i:4d}/{end_idx-1} [{progress_pct:5.1f}%] ", end='', flush=True)

            success = self.execute_action(action, mode=mode)

            if success:
                executed += 1
            else:
                failed += 1
                print(f"\n  ⚠️  Action {i} FAILED")

            if delay > 0:
                time.sleep(delay)

        print(f"\n\n{'='*60}")
        print(f"✓ Completed: {executed}/{num_actions} successful, {failed} failed")
        print(f"{'='*60}")
        return executed

    def go_to_home(self):
        """Move arm to home pose."""
        self.bot.arm.go_to_home_pose(blocking=True)

    def go_to_sleep(self):
        """Move arm to sleep pose."""
        self.bot.arm.go_to_sleep_pose(blocking=True)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Execute actions from JSON on a single VX300S arm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute left arm joint actions
  python3 execute_json_actions.py joints_episode_actions_2.json --arm left --mode joint

  # Execute right arm EE actions
  python3 execute_json_actions.py ee_actions.json --arm right --mode ee --max-actions 100

  # Start from action 500 with slower timing
  python3 execute_json_actions.py actions.json --arm left --start-idx 500 --moving-time 1.5
        """
    )

    parser.add_argument(
        'json_file',
        type=str,
        help='Path to JSON file containing actions'
    )
    parser.add_argument(
        '--arm',
        type=str,
        required=True,
        choices=['left', 'right'],
        help='Which arm to control (required)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='joint',
        choices=['joint', 'ee'],
        help='Control mode: joint or end-effector (default: joint)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting action index (default: 0)'
    )
    parser.add_argument(
        '--max-actions',
        type=int,
        default=None,
        help='Maximum number of actions to execute (default: all)'
    )
    parser.add_argument(
        '--moving-time',
        type=float,
        default=1.0,
        help='Time per action in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--accel-time',
        type=float,
        default=0.3,
        help='Acceleration time in seconds (default: 0.3)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='Additional delay between actions in seconds (default: 0.0)'
    )
    parser.add_argument(
        '--skip-home',
        action='store_true',
        help='Skip going to home pose at start'
    )
    parser.add_argument(
        '--skip-sleep',
        action='store_true',
        help='Skip going to sleep pose at end'
    )

    args = parser.parse_args()

    # Validate JSON file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    # Initialize executor
    executor = SingleArmExecutor(
        arm_side=args.arm,
        moving_time=args.moving_time,
        accel_time=args.accel_time,
    )

    # Start robot
    robot_startup()

    try:
        # Load actions
        actions, metadata = executor.load_json_actions(str(json_path))

        # Go to home
        if not args.skip_home:
            print("\nMoving to home pose...")
            executor.go_to_home()
            time.sleep(1.0)

        # Execute trajectory
        print("\nStarting execution...")
        start_time = time.time()

        num_executed = executor.execute_trajectory(
            actions=actions,
            mode=args.mode,
            start_idx=args.start_idx,
            max_actions=args.max_actions,
            delay=args.delay,
        )

        elapsed_time = time.time() - start_time

        print(f"\nExecution completed in {elapsed_time:.2f} seconds")
        if num_executed > 0:
            print(f"Average time per action: {elapsed_time/num_executed:.3f} seconds")

        # Go to sleep
        if not args.skip_sleep:
            print("\nMoving to sleep pose...")
            time.sleep(1.0)
            executor.go_to_sleep()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Moving to sleep pose...")
        executor.go_to_sleep()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        robot_shutdown()
        print("Shutdown complete")

    return 0


if __name__ == '__main__':
    exit(main())
