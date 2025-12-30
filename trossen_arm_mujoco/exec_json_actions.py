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
"""

import argparse
import json
import time

import numpy as np

from trossen_arm_mujoco.constants import BOX_POSE
from trossen_arm_mujoco.sim_env import TransferCubeTask
from trossen_arm_mujoco.utils import (
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
)


# JSON action indices
# right_waist=0, right_shoulder=1, right_elbow=2, right_forearm_roll=3,
# right_wrist_angle=4, right_wrist_rotate=5, right_gripper=6,
# left_waist=7, left_shoulder=8, left_elbow=9, left_forearm_roll=10,
# left_wrist_angle=11, left_wrist_rotate=12, left_gripper=13

# Simulation expects action format:
# [left_arm(6), left_gripper(1), unused(1), right_arm(6), right_gripper(1), unused(1)]
# left_arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
# right_arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate


def reorder_action(json_action: np.ndarray) -> np.ndarray:
    """
    Reorder action from JSON format to simulation format.

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
    :return: Action array in simulation format (16 values).
    """
    # Extract components from JSON action
    right_arm = json_action[0:6]  # right_waist to right_wrist_rotate
    right_gripper = json_action[6]
    left_arm = json_action[7:13]  # left_waist to left_wrist_rotate
    left_gripper = json_action[13]

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


def load_actions_from_json(json_path: str) -> np.ndarray:
    """
    Load actions from a JSON file.

    :param json_path: Path to the JSON file containing actions.
    :return: NumPy array of actions with shape (num_steps, 14).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    actions = np.array(data["actions"], dtype=np.float32)

    # Handle shape [num_steps, 1, 14] -> [num_steps, 14]
    if len(actions.shape) == 3 and actions.shape[1] == 1:
        actions = actions.squeeze(axis=1)

    print(f"Loaded {len(actions)} actions from {json_path}")
    print(f"Actions shape: {actions.shape}")

    return actions


def execute_actions(
    json_path: str,
    onscreen_render: bool = True,
    cam_list: list[str] | None = None,
    playback_speed: float = 1.0,
) -> list:
    """
    Execute actions from a JSON file in the simulation.

    :param json_path: Path to the JSON file containing actions.
    :param onscreen_render: Whether to render the simulation on-screen.
    :param cam_list: List of cameras for observation capture.
    :param playback_speed: Speed multiplier for playback (1.0 = real-time).
    :return: List of timesteps from the episode.
    """
    if cam_list is None:
        cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    # Load actions from JSON
    actions = load_actions_from_json(json_path)

    # Create the simulation environment
    env = make_sim_env(
        TransferCubeTask,
        xml_file="trossen_ai_scene_joint.xml",
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

    # Execute each action
    for t, json_action in enumerate(actions):
        # Reorder action from JSON format to simulation format
        sim_action = reorder_action(json_action)

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

    args = parser.parse_args()

    execute_actions(
        json_path=args.json_path,
        onscreen_render=not args.no_render,
        cam_list=args.cameras,
        playback_speed=args.speed,
    )


if __name__ == "__main__":
    main()
