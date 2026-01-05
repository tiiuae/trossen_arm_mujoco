# Action Format for exec_json_actions.py

This document describes the action input format and units expected by the `step` function in the MuJoCo simulation.

---

## Robot Arm Joint Diagram

```
                                    GRIPPER
                                   ┌───┬───┐
                                   │   │   │  ← Gripper fingers
                                   └─┬─┴─┬─┘
                                     │   │
                                   ══╪═══╪══  ← WRIST_ROTATE (rotation around forearm axis)
                                     └─┬─┘
                                       │
                                    ┌──┴──┐
                                    │     │   ← WRIST_ANGLE (up/down pitch)
                                    └──┬──┘
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║            FOREARM                  ║
                    ╚══════════════════╤══════════════════╝
                                       │
                                   ════╪════  ← FOREARM_ROLL (rotation around upper arm axis)
                                       │
                                    ┌──┴──┐
                                    │     │   ← ELBOW (up/down pitch)
                                    └──┬──┘
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║            UPPER ARM                ║
                    ╚══════════════════╤══════════════════╝
                                       │
                                    ┌──┴──┐
                                    │     │   ← SHOULDER (up/down pitch)
                                    └──┬──┘
                                       │
                               ════════╪════════  ← WAIST (rotation around vertical axis)
                                       │
                              ╔════════╧════════╗
                              ║      BASE       ║
                              ╚═════════════════╝
                             ////////////////////
                                   GROUND
```

### Joint Rotation Axes

| Joint | Axis | Motion |
|-------|------|--------|
| **waist** | Z (vertical) | Rotate left/right |
| **shoulder** | Y (horizontal) | Pitch up/down |
| **elbow** | Y (horizontal) | Pitch up/down |
| **forearm_roll** | X (along arm) | Roll forearm |
| **wrist_angle** | Y (horizontal) | Pitch up/down |
| **wrist_rotate** | X (along arm) | Roll gripper |
| **gripper** | - | Open/close (linear) |

---

## Joint Mode (`--mode joint`)

The simulation expects **16 values** in this format:

| Index | Value | Units |
|-------|-------|-------|
| 0-5 | Left arm joints (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate) | **radians** |
| 6 | Left gripper | **meters** (gripper opening) |
| 7 | Unused | - |
| 8-13 | Right arm joints (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate) | **radians** |
| 14 | Right gripper | **meters** (gripper opening) |
| 15 | Unused | - |

### JSON Input Format (14 values)

The JSON file uses a different ordering that gets converted by `reorder_action()`:

```
[right_waist, right_shoulder, right_elbow, right_forearm_roll,
 right_wrist_angle, right_wrist_rotate, right_gripper,
 left_waist, left_shoulder, left_elbow, left_forearm_roll,
 left_wrist_angle, left_wrist_rotate, left_gripper]
```

---

## End Effector Mode (`--mode ee`)

The simulation expects **16 values** in this format:

| Index | Value | Units |
|-------|-------|-------|
| 0-2 | Left EE position (x, y, z) | **meters** |
| 3-6 | Left EE orientation (qw, qx, qy, qz) | **quaternion** (unitless) |
| 7 | Left gripper | **meters** |
| 8-10 | Right EE position (x, y, z) | **meters** |
| 11-14 | Right EE orientation (qw, qx, qy, qz) | **quaternion** (unitless) |
| 15 | Right gripper | **meters** |

### JSON Input Format (14 values)

The JSON file uses Euler angles that get converted to quaternions by `reorder_action_ee()`:

```
[right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper,
 left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper]
```

- Position values (x, y, z): **meters**
- Euler angles (roll, pitch, yaw): **radians**
- Gripper: **meters**

---

## Key Constants

### Home Joint Positions
```python
HOME_ARM_JOINTS = [0.0, π/12, π/12, 0.0, 0.0, 0.0]  # radians
```

### Home Gripper Value
```python
HOME_GRIPPER = 0.044  # meters
```

### Home End Effector Positions
```python
HOME_LEFT_EE_POS = [-0.19657, -0.019, 0.25021]   # meters (x, y, z)
HOME_RIGHT_EE_POS = [0.19657, -0.019, 0.25021]   # meters (x, y, z) - mirrored X
HOME_EE_QUAT = [1.0, 0.0, 0.0, 0.0]              # quaternion (qw, qx, qy, qz)
```

---

## Notes

- **Euler to Quaternion Conversion**: Uses XYZ (sxyz) convention to match Interbotix's `angle_manipulation` module.
- **Arm Selection**: The `--arms` flag allows moving only `left`, `right`, or `both` arms.
- **Unused indices**: Indices 7 and 15 in joint mode are unused padding values (set to 0.0).
