# Copyright 2025 Trossen Robotics
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
Execute actions from a JSON file in the MuJoCo simulation.

Supports two modes:
1. Joint position mode (default):
   JSON format expected:
   {
       "shape": [num_steps, 1, 14],
       "dtype": "float32",
       "actions": [[[action_0], [action_1], ...]]
   }

   Action order in JSON (14 values):
       right_waist, right_shoulder, right_elbow, right_forearm_roll,
       right_wrist_angle, right_wrist_rotate, right_gripper,
       left_waist, left_shoulder, left_elbow, left_forearm_roll,
       left_wrist_angle, left_wrist_rotate, left_gripper

2. End effector (EE) position mode:
   JSON format expected (from joint_to_ee converter):
   {
       "shape": [num_steps, 1, 14],
       "dtype": "float32",
       "actions": [[[action_0], [action_1], ...]]
   }

   Action order in JSON (14 values):
       right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper,
       left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper

   Note: Euler angles are converted to quaternions for the simulation.
"""

import argparse
import json
import time

import numpy as np

from trossen_arm_mujoco.constants import BOX_POSE, START_ARM_POSE
from trossen_arm_mujoco.ee_sim_env import TransferCubeEETask
from trossen_arm_mujoco.sim_env import TransferCubeTask
from trossen_arm_mujoco.utils import (
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
)


# JSON action indices for joint mode
# right_waist=0, right_shoulder=1, right_elbow=2, right_forearm_roll=3,
# right_wrist_angle=4, right_wrist_rotate=5, right_gripper=6,
# left_waist=7, left_shoulder=8, left_elbow=9, left_forearm_roll=10,
# left_wrist_angle=11, left_wrist_rotate=12, left_gripper=13

# Simulation expects joint action format:
# [left_arm(6), left_gripper(1), unused(1), right_arm(6), right_gripper(1), unused(1)]
# left_arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
# right_arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate

# JSON action indices for EE mode (from joint_to_ee converter)
# right_x=0, right_y=1, right_z=2, right_roll=3, right_pitch=4, right_yaw=5, right_gripper=6,
# left_x=7, left_y=8, left_z=9, left_roll=10, left_pitch=11, left_yaw=12, left_gripper=13

# Simulation expects EE action format:
# [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
#  right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz).
    Uses XYZ (sxyz) convention to match Interbotix's angle_manipulation module.

    :param roll: Roll angle in radians.
    :param pitch: Pitch angle in radians.
    :param yaw: Yaw angle in radians.
    :return: Quaternion array [qw, qx, qy, qz].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


# Home position joint values from START_ARM_POSE: [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
# START_ARM_POSE format: [left_arm(6), left_gripper(2), right_arm(6), right_gripper(2)]
# Home position is [0.0, π/12, π/12, 0.0, 0.0, 0.0] for 6 joints
HOME_ARM_JOINTS = np.array(START_ARM_POSE[:6])  # First 6 values are left arm home position
HOME_GRIPPER = START_ARM_POSE[6]  # Gripper value (0.044)

# Home EE pose (from ee_sim_env.py mocap reset position - this is the EE position when arm is at home)
# Left and right arms have mirrored X positions
HOME_LEFT_EE_POS = np.array([-0.19657, -0.019, 0.25021])  # x, y, z
HOME_RIGHT_EE_POS = np.array([0.19657, -0.019, 0.25021])  # x, y, z (mirrored X)
HOME_EE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # qw, qx, qy, qz


def reorder_action(json_action: np.ndarray, arms: str = "both") -> np.ndarray:
    """
    Reorder action from JSON format to simulation format (joint mode).

    JSON format (14 values):
        [right_waist, right_shoulder, right_elbow, right_forearm_roll,
         right_wrist_angle, right_wrist_rotate, right_gripper,
         left_waist, left_shoulder, left_elbow, left_forearm_roll,
         left_wrist_angle, left_wrist_rotate, left_gripper]

    Simulation format (16 values):
        [left_waist, left_shoulder, left_elbow, left_forearm_roll,
         left_wrist_angle, left_wrist_rotate, left_gripper, unused,
         right_waist, right_shoulder, right_elbow, right_forearm_roll,
         right_wrist_angle, right_wrist_rotate, right_gripper, unused]

    :param json_action: Action array in JSON format (14 values).
    :param arms: Which arms to move - "left", "right", or "both".
    :return: Action array in simulation format (16 values).
    """
    # Extract components from JSON action
    right_arm = json_action[0:6]  # right_waist to right_wrist_rotate
    right_gripper = json_action[6]
    left_arm = json_action[7:13]  # left_waist to left_wrist_rotate
    left_gripper = json_action[13]

    # Apply arm filtering
    if arms == "left":
        right_arm = HOME_ARM_JOINTS
        right_gripper = HOME_GRIPPER
    elif arms == "right":
        left_arm = HOME_ARM_JOINTS
        left_gripper = HOME_GRIPPER

    # Reorder for simulation: left arm, left gripper, unused, right arm, right gripper, unused
    sim_action = np.concatenate([
        left_arm,           # indices 0-5
        [left_gripper],     # index 6
        [0.0],              # index 7 (unused)
        right_arm,          # indices 8-13
        [right_gripper],    # index 14
        [0.0],              # index 15 (unused)
    ])

    return sim_action


def reorder_action_ee(json_action: np.ndarray, arms: str = "both") -> np.ndarray:
    """
    Reorder action from JSON format to simulation format (EE mode).

    JSON format from joint_to_ee converter (14 values):
        [right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper,
         left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper]

    Simulation format (16 values):
        [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
         right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]

    :param json_action: Action array in JSON format (14 values with Euler angles).
    :param arms: Which arms to move - "left", "right", or "both".
    :return: Action array in simulation format (16 values with quaternions).
    """
    # Extract components from JSON action
    # Right arm: position (0-2), euler angles (3-5), gripper (6)
    right_pos = json_action[0:3]
    right_euler = json_action[3:6]  # roll, pitch, yaw
    right_gripper = json_action[6]

    # Left arm: position (7-9), euler angles (10-12), gripper (13)
    left_pos = json_action[7:10]
    left_euler = json_action[10:13]  # roll, pitch, yaw
    left_gripper = json_action[13]

    # Convert Euler angles to quaternions
    right_quat = euler_to_quaternion(right_euler[0], right_euler[1], right_euler[2])
    left_quat = euler_to_quaternion(left_euler[0], left_euler[1], left_euler[2])

    # Apply arm filtering
    if arms == "left":
        right_pos = HOME_RIGHT_EE_POS.copy()
        right_quat = HOME_EE_QUAT.copy()
        right_gripper = HOME_GRIPPER
    elif arms == "right":
        left_pos = HOME_LEFT_EE_POS.copy()
        left_quat = HOME_EE_QUAT.copy()
        left_gripper = HOME_GRIPPER

    # Reorder for simulation: left pose, left gripper, right pose, right gripper
    sim_action = np.concatenate([
        left_pos,           # indices 0-2 (x, y, z)
        left_quat,          # indices 3-6 (qw, qx, qy, qz)
        [left_gripper],     # index 7
        right_pos,          # indices 8-10 (x, y, z)
        right_quat,         # indices 11-14 (qw, qx, qy, qz)
        [right_gripper],    # index 15
    ])

    return sim_action


def load_actions_from_json(json_path: str, mode: str = "joint") -> np.ndarray:
    """
    Load actions from a JSON file.

    :param json_path: Path to the JSON file containing actions.
    :param mode: Action mode - "joint" (14 values) or "ee" (16 values).
    :return: NumPy array of actions.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    actions = np.array(data["actions"], dtype=np.float32)

    # Handle shape [num_steps, 1, N] -> [num_steps, N]
    if len(actions.shape) == 3 and actions.shape[1] == 1:
        actions = actions.squeeze(axis=1)

    # Both joint and ee modes expect 14 action dimensions
    expected_dim = 14
    if actions.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} action dimensions for '{mode}' mode, "
            f"but got {actions.shape[1]}. Check your JSON file or mode selection."
        )

    print(f"Loaded {len(actions)} actions from {json_path}")
    print(f"Actions shape: {actions.shape}")
    print(f"Mode: {mode}")

    return actions


def execute_actions(
    json_path: str,
    onscreen_render: bool = True,
    cam_list: list[str] | None = None,
    playback_speed: float = 1.0,
    mode: str = "joint",
    arms: str = "both",
) -> list:
    """
    Execute actions from a JSON file in the simulation.

    :param json_path: Path to the JSON file containing actions.
    :param onscreen_render: Whether to render the simulation on-screen.
    :param cam_list: List of cameras for observation capture.
    :param playback_speed: Speed multiplier for playback (1.0 = real-time).
    :param mode: Action mode - "joint" for joint positions or "ee" for end effector positions.
    :param arms: Which arms to move - "left", "right", or "both" (default).
    :return: List of timesteps from the episode.
    """
    if cam_list is None:
        cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    # Load actions from JSON
    actions = load_actions_from_json(json_path, mode=mode)

    # Select task class and XML file based on mode
    if mode == "joint":
        task_class = TransferCubeTask
        xml_file = "trossen_ai_scene_joint.xml"
        reorder_fn = reorder_action
    elif mode == "ee":
        task_class = TransferCubeEETask
        xml_file = "trossen_ai_scene.xml"
        reorder_fn = reorder_action_ee
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'joint' or 'ee'.")

    # Create the simulation environment
    env = make_sim_env(
        task_class,
        xml_file=xml_file,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
    )

    # Set box pose before reset (required by TransferCubeTask)
    BOX_POSE[0] = sample_box_pose()

    # Reset the environment
    ts = env.reset()
    episode = [ts]

    # Setup plotting if rendering
    plt_imgs = None
    if onscreen_render:
        plt_imgs = plot_observation_images(ts.observation, cam_list)

    # Calculate delay between steps for playback speed
    dt = 0.02  # Default simulation timestep
    step_delay = dt / playback_speed

    print(f"Executing {len(actions)} actions...")
    print(f"Playback speed: {playback_speed}x")
    print(f"Using {mode} mode with {xml_file}")
    print(f"Arms: {arms}")

    # Execute each action
    for t, json_action in enumerate(actions):
        # Reorder action from JSON format to simulation format
        sim_action = reorder_fn(json_action, arms=arms)

        # Step the simulation
        ts = env.step(sim_action)
        episode.append(ts)

        # Update visualization
        if onscreen_render and plt_imgs is not None:
            plt_imgs = set_observation_images(ts.observation, plt_imgs, cam_list)

        # Add delay for playback speed control
        if playback_speed < 10.0:  # Only add delay if not at maximum speed
            time.sleep(step_delay)

        # Print progress every 100 steps
        if (t + 1) % 100 == 0:
            print(f"Step {t + 1}/{len(actions)}")

    print(f"Finished executing {len(actions)} actions")

    return episode


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Execute actions from a JSON file in the MuJoCo simulation."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the JSON file containing actions.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable on-screen rendering.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 for real-time).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of cameras to use for observations.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["joint", "ee"],
        default="joint",
        help="Action mode: 'joint' for joint positions (14 values) or 'ee' for end effector positions (16 values). Default: joint.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which arms to move: 'left', 'right', or 'both'. Default: both.",
    )

    args = parser.parse_args()

    execute_actions(
        json_path=args.json_path,
        onscreen_render=not args.no_render,
        cam_list=args.cameras,
        playback_speed=args.speed,
        mode=args.mode,
        arms=args.arms,
    )


if __name__ == "__main__":
    main()
