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
Interactive joint control for the MuJoCo simulation using sliders.

This script provides a GUI with sliders to control each joint angle
of both the left and right arms in real-time.

Simulation action format (16 values):
    [left_waist, left_shoulder, left_elbow, left_forearm_roll,
     left_wrist_angle, left_wrist_rotate, left_gripper, unused,
     right_waist, right_shoulder, right_elbow, right_forearm_roll,
     right_wrist_angle, right_wrist_rotate, right_gripper, unused]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from trossen_arm_mujoco.constants import BOX_POSE
from trossen_arm_mujoco.sim_env import TransferCubeTask
from trossen_arm_mujoco.utils import (
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
)


# Joint configuration
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
]

# Joint limits (in radians, approximate values for wxai arm)
JOINT_LIMITS = {
    "waist": (-np.pi, np.pi),
    "shoulder": (-np.pi/2, np.pi/2),
    "elbow": (-np.pi/2, np.pi/2),
    "forearm_roll": (-np.pi, np.pi),
    "wrist_angle": (-np.pi/2, np.pi/2),
    "wrist_rotate": (-np.pi, np.pi),
    "gripper": (0.0, 0.044),
}

# Default starting pose (from constants.py)
DEFAULT_POSE = {
    "waist": 0.0,
    "shoulder": np.pi / 12,
    "elbow": np.pi / 12,
    "forearm_roll": 0.0,
    "wrist_angle": 0.0,
    "wrist_rotate": 0.0,
    "gripper": 0.044,
}


class InteractiveJointController:
    """Interactive controller for robot joint angles using sliders."""

    def __init__(
        self,
        onscreen_render: bool = True,
        cam_list: list[str] | None = None,
    ):
        """
        Initialize the interactive joint controller.

        :param onscreen_render: Whether to render the simulation on-screen.
        :param cam_list: List of cameras for observation capture.
        """
        if cam_list is None:
            cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        self.cam_list = cam_list
        self.onscreen_render = onscreen_render

        # Current joint values for left and right arms
        self.left_joints = {name: DEFAULT_POSE[name] for name in JOINT_NAMES}
        self.right_joints = {name: DEFAULT_POSE[name] for name in JOINT_NAMES}

        # Create the simulation environment
        self.env = make_sim_env(
            TransferCubeTask,
            xml_file="trossen_ai_scene_joint.xml",
            task_name="sim_transfer_cube",
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )

        # Set box pose before reset
        BOX_POSE[0] = sample_box_pose()

        # Reset the environment
        self.ts = self.env.reset()

        # Setup observation plotting
        self.plt_imgs = None
        if onscreen_render:
            self.plt_imgs = plot_observation_images(self.ts.observation, cam_list)

        # Create slider GUI
        self._create_slider_gui()

    def _build_action(self) -> np.ndarray:
        """
        Build the simulation action from current joint values.

        :return: Action array in simulation format (16 values).
        """
        # Build left arm action
        left_arm = np.array([
            self.left_joints["waist"],
            self.left_joints["shoulder"],
            self.left_joints["elbow"],
            self.left_joints["forearm_roll"],
            self.left_joints["wrist_angle"],
            self.left_joints["wrist_rotate"],
        ])
        left_gripper = self.left_joints["gripper"]

        # Build right arm action
        right_arm = np.array([
            self.right_joints["waist"],
            self.right_joints["shoulder"],
            self.right_joints["elbow"],
            self.right_joints["forearm_roll"],
            self.right_joints["wrist_angle"],
            self.right_joints["wrist_rotate"],
        ])
        right_gripper = self.right_joints["gripper"]

        # Combine into simulation format
        sim_action = np.concatenate([
            left_arm,           # indices 0-5
            [left_gripper],     # index 6
            [0.0],              # index 7 (unused)
            right_arm,          # indices 8-13
            [right_gripper],    # index 14
            [0.0],              # index 15 (unused)
        ])

        return sim_action

    def _update_simulation(self):
        """Update the simulation with current joint values."""
        action = self._build_action()
        self.ts = self.env.step(action)

        # Update observation images
        if self.onscreen_render and self.plt_imgs is not None:
            self.plt_imgs = set_observation_images(
                self.ts.observation, self.plt_imgs, self.cam_list
            )

    def _create_slider_callback(self, arm: str, joint_name: str):
        """
        Create a callback function for a slider.

        :param arm: 'left' or 'right'.
        :param joint_name: Name of the joint.
        :return: Callback function.
        """
        def callback(val):
            if arm == "left":
                self.left_joints[joint_name] = val
            else:
                self.right_joints[joint_name] = val
            self._update_simulation()

        return callback

    def _reset_callback(self, event):
        """Reset all sliders to default values."""
        for name in JOINT_NAMES:
            self.left_joints[name] = DEFAULT_POSE[name]
            self.right_joints[name] = DEFAULT_POSE[name]

        # Reset all sliders
        for name in JOINT_NAMES:
            self.left_sliders[name].set_val(DEFAULT_POSE[name])
            self.right_sliders[name].set_val(DEFAULT_POSE[name])

        self._update_simulation()

    def _create_slider_gui(self):
        """Create the slider GUI for joint control."""
        # Create figure for sliders
        self.slider_fig, self.slider_axes = plt.subplots(figsize=(14, 10))
        self.slider_fig.canvas.manager.set_window_title("Joint Control Sliders")

        # Hide the main axes
        self.slider_axes.set_visible(False)

        # Calculate slider positions
        slider_height = 0.03
        slider_width = 0.35
        vertical_spacing = 0.055

        left_x = 0.08
        right_x = 0.55
        start_y = 0.85

        # Create title
        self.slider_fig.text(0.27, 0.95, "Left Arm", fontsize=14, fontweight="bold", ha="center")
        self.slider_fig.text(0.73, 0.95, "Right Arm", fontsize=14, fontweight="bold", ha="center")

        # Store sliders
        self.left_sliders = {}
        self.right_sliders = {}

        # Create sliders for each joint
        for i, joint_name in enumerate(JOINT_NAMES):
            y_pos = start_y - i * vertical_spacing
            limits = JOINT_LIMITS[joint_name]

            # Left arm slider
            left_ax = self.slider_fig.add_axes([left_x, y_pos, slider_width, slider_height])
            left_slider = Slider(
                left_ax,
                f"L {joint_name}",
                limits[0],
                limits[1],
                valinit=DEFAULT_POSE[joint_name],
                valstep=0.01,
            )
            left_slider.on_changed(self._create_slider_callback("left", joint_name))
            self.left_sliders[joint_name] = left_slider

            # Right arm slider
            right_ax = self.slider_fig.add_axes([right_x, y_pos, slider_width, slider_height])
            right_slider = Slider(
                right_ax,
                f"R {joint_name}",
                limits[0],
                limits[1],
                valinit=DEFAULT_POSE[joint_name],
                valstep=0.01,
            )
            right_slider.on_changed(self._create_slider_callback("right", joint_name))
            self.right_sliders[joint_name] = right_slider

        # Add reset button
        reset_ax = self.slider_fig.add_axes([0.4, 0.02, 0.2, 0.04])
        self.reset_button = Button(reset_ax, "Reset to Default", hovercolor="0.8")
        self.reset_button.on_clicked(self._reset_callback)

        # Add instructions
        self.slider_fig.text(
            0.5, 0.08,
            "Adjust sliders to control joint angles. Close this window to exit.",
            fontsize=10, ha="center", style="italic"
        )

        # Print current values
        self._print_current_values()

    def _print_current_values(self):
        """Print current joint values to console."""
        print("\n" + "=" * 60)
        print("Current Joint Values:")
        print("-" * 60)
        print(f"{'Joint':<15} {'Left':>12} {'Right':>12}")
        print("-" * 60)
        for name in JOINT_NAMES:
            print(f"{name:<15} {self.left_joints[name]:>12.4f} {self.right_joints[name]:>12.4f}")
        print("=" * 60)

    def run(self):
        """Run the interactive controller."""
        print("\nInteractive Joint Controller Started!")
        print("Adjust the sliders to control the robot joints.")
        print("Close the slider window to exit.\n")

        # Use interactive mode and manual event loop to keep window open
        plt.ion()
        self.slider_fig.show()

        # Keep the script running while the slider window is open
        try:
            while plt.fignum_exists(self.slider_fig.number):
                plt.pause(0.1)
        except KeyboardInterrupt:
            pass

        print("\nController closed.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Interactive joint control for the MuJoCo simulation."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable on-screen rendering.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of cameras to use for observations.",
    )

    args = parser.parse_args()

    controller = InteractiveJointController(
        onscreen_render=not args.no_render,
        cam_list=args.cameras,
    )
    controller.run()


if __name__ == "__main__":
    main()
