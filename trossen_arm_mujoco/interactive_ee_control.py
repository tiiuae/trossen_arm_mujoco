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
Interactive end effector control for the MuJoCo simulation using sliders.

This script provides a GUI with sliders to control the end effector position,
orientation (quaternion), and gripper of both the left and right arms in real-time.

Simulation action format (16 values):
    [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
     right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from trossen_arm_mujoco.constants import BOX_POSE
from trossen_arm_mujoco.ee_sim_env import TransferCubeEETask
from trossen_arm_mujoco.utils import (
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
)


# End effector parameters
EE_PARAMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# Parameter limits
EE_LIMITS = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.0, 0.6),
    "roll": (-np.pi, np.pi),
    "pitch": (-np.pi, np.pi),
    "yaw": (-np.pi, np.pi),
    "gripper": (0.0, 0.044),
}

# Default starting pose (from ee_sim_env.py initialize_robots)
DEFAULT_LEFT_POSE = {
    "x": -0.19657,
    "y": -0.019,
    "z": 0.25021,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "gripper": 0.044,
}

DEFAULT_RIGHT_POSE = {
    "x": 0.19657,
    "y": -0.019,
    "z": 0.25021,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "gripper": 0.044,
}


class InteractiveEEController:
    """Interactive controller for robot end effector poses using sliders."""

    def __init__(
        self,
        onscreen_render: bool = True,
        cam_list: list[str] | None = None,
    ):
        """
        Initialize the interactive end effector controller.

        :param onscreen_render: Whether to render the simulation on-screen.
        :param cam_list: List of cameras for observation capture.
        """
        if cam_list is None:
            cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        self.cam_list = cam_list
        self.onscreen_render = onscreen_render

        # Current EE values for left and right arms
        self.left_ee = {name: DEFAULT_LEFT_POSE[name] for name in EE_PARAMS}
        self.right_ee = {name: DEFAULT_RIGHT_POSE[name] for name in EE_PARAMS}

        # Create the simulation environment
        self.env = make_sim_env(
            TransferCubeEETask,
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

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> tuple:
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).

        Uses the ZYX (yaw-pitch-roll) convention.

        :param roll: Rotation around X axis (radians).
        :param pitch: Rotation around Y axis (radians).
        :param yaw: Rotation around Z axis (radians).
        :return: Quaternion as (qw, qx, qy, qz).
        """
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qw, qx, qy, qz

    def _build_action(self) -> np.ndarray:
        """
        Build the simulation action from current end effector values.

        :return: Action array in simulation format (16 values).
        """
        # Convert Euler angles to quaternions
        left_qw, left_qx, left_qy, left_qz = self._euler_to_quaternion(
            self.left_ee["roll"], self.left_ee["pitch"], self.left_ee["yaw"]
        )
        right_qw, right_qx, right_qy, right_qz = self._euler_to_quaternion(
            self.right_ee["roll"], self.right_ee["pitch"], self.right_ee["yaw"]
        )

        # Build action array
        # Format: [left_pos(3), left_quat(4), left_gripper(1), right_pos(3), right_quat(4), right_gripper(1)]
        sim_action = np.array([
            # Left arm
            self.left_ee["x"],
            self.left_ee["y"],
            self.left_ee["z"],
            left_qw,
            left_qx,
            left_qy,
            left_qz,
            self.left_ee["gripper"],
            # Right arm
            self.right_ee["x"],
            self.right_ee["y"],
            self.right_ee["z"],
            right_qw,
            right_qx,
            right_qy,
            right_qz,
            self.right_ee["gripper"],
        ])

        return sim_action

    def _update_simulation(self):
        """Update the simulation with current end effector values."""
        action = self._build_action()
        self.ts = self.env.step(action)

        # Update observation images
        if self.onscreen_render and self.plt_imgs is not None:
            self.plt_imgs = set_observation_images(
                self.ts.observation, self.plt_imgs, self.cam_list
            )

    def _create_slider_callback(self, arm: str, param_name: str):
        """
        Create a callback function for a slider.

        :param arm: 'left' or 'right'.
        :param param_name: Name of the parameter.
        :return: Callback function.
        """
        def callback(val):
            if arm == "left":
                self.left_ee[param_name] = val
            else:
                self.right_ee[param_name] = val
            self._update_simulation()

        return callback

    def _reset_callback(self, event):
        """Reset all sliders to default values."""
        for name in EE_PARAMS:
            self.left_ee[name] = DEFAULT_LEFT_POSE[name]
            self.right_ee[name] = DEFAULT_RIGHT_POSE[name]

        # Reset all sliders
        for name in EE_PARAMS:
            self.left_sliders[name].set_val(DEFAULT_LEFT_POSE[name])
            self.right_sliders[name].set_val(DEFAULT_RIGHT_POSE[name])

        self._update_simulation()

    def _create_slider_gui(self):
        """Create the slider GUI for end effector control."""
        # Create figure for sliders
        self.slider_fig, self.slider_axes = plt.subplots(figsize=(14, 12))
        self.slider_fig.canvas.manager.set_window_title("End Effector Control Sliders")

        # Hide the main axes
        self.slider_axes.set_visible(False)

        # Calculate slider positions
        slider_height = 0.03
        slider_width = 0.35
        vertical_spacing = 0.05

        left_x = 0.08
        right_x = 0.55
        start_y = 0.88

        # Create title
        self.slider_fig.text(0.27, 0.95, "Left Arm EE", fontsize=14, fontweight="bold", ha="center")
        self.slider_fig.text(0.73, 0.95, "Right Arm EE", fontsize=14, fontweight="bold", ha="center")

        # Add section labels
        self.slider_fig.text(0.02, 0.90, "Position", fontsize=10, fontweight="bold", va="center", rotation=90)
        self.slider_fig.text(0.02, 0.65, "Orientation", fontsize=10, fontweight="bold", va="center", rotation=90)
        self.slider_fig.text(0.02, 0.40, "Gripper", fontsize=10, fontweight="bold", va="center", rotation=90)

        # Store sliders
        self.left_sliders = {}
        self.right_sliders = {}

        # Create sliders for each parameter
        for i, param_name in enumerate(EE_PARAMS):
            y_pos = start_y - i * vertical_spacing
            limits = EE_LIMITS[param_name]

            # Determine step size based on parameter type
            if param_name in ["x", "y", "z"]:
                step = 0.005
            elif param_name == "gripper":
                step = 0.001
            else:
                step = 0.01

            # Left arm slider
            left_ax = self.slider_fig.add_axes([left_x, y_pos, slider_width, slider_height])
            left_slider = Slider(
                left_ax,
                f"L {param_name}",
                limits[0],
                limits[1],
                valinit=DEFAULT_LEFT_POSE[param_name],
                valstep=step,
            )
            left_slider.on_changed(self._create_slider_callback("left", param_name))
            self.left_sliders[param_name] = left_slider

            # Right arm slider
            right_ax = self.slider_fig.add_axes([right_x, y_pos, slider_width, slider_height])
            right_slider = Slider(
                right_ax,
                f"R {param_name}",
                limits[0],
                limits[1],
                valinit=DEFAULT_RIGHT_POSE[param_name],
                valstep=step,
            )
            right_slider.on_changed(self._create_slider_callback("right", param_name))
            self.right_sliders[param_name] = right_slider

        # Add reset button
        reset_ax = self.slider_fig.add_axes([0.4, 0.02, 0.2, 0.04])
        self.reset_button = Button(reset_ax, "Reset to Default", hovercolor="0.8")
        self.reset_button.on_clicked(self._reset_callback)

        # Add instructions
        self.slider_fig.text(
            0.5, 0.08,
            "Adjust sliders to control end effector pose. Roll/Pitch/Yaw in radians.\nClose this window to exit.",
            fontsize=10, ha="center", style="italic"
        )

        # Print current values
        self._print_current_values()

    def _print_current_values(self):
        """Print current end effector values to console."""
        print("\n" + "=" * 70)
        print("Current End Effector Values:")
        print("-" * 70)
        print(f"{'Parameter':<15} {'Left':>15} {'Right':>15}")
        print("-" * 70)
        for name in EE_PARAMS:
            print(f"{name:<15} {self.left_ee[name]:>15.5f} {self.right_ee[name]:>15.5f}")
        print("=" * 70)

    def run(self):
        """Run the interactive controller."""
        print("\nInteractive End Effector Controller Started!")
        print("Adjust the sliders to control the robot end effectors.")
        print("Orientation is controlled via Roll/Pitch/Yaw (radians).")
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
        description="Interactive end effector control for the MuJoCo simulation."
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

    controller = InteractiveEEController(
        onscreen_render=not args.no_render,
        cam_list=args.cameras,
    )
    controller.run()


if __name__ == "__main__":
    main()
