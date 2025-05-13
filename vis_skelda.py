import sys

import numpy as np

sys.path.append("/PoseForecasters/")
import utils_show

# ==================================================================================================


def visualize(input_seq, output_seq):

    dim = input_seq.shape[-1]
    poses_input = input_seq[0, 0].cpu().numpy().reshape(-1, dim // 3, 3)
    poses_target = output_seq[0, 0].cpu().numpy().reshape(-1, dim // 3, 3)
    poses_input = poses_input[:, :, [0, 2, 1]]
    poses_target = poses_target[:, :, [0, 2, 1]]

    if dim == 45:
        joint_names = [
            "hip_middle",
            "hip_right",
            "knee_right",
            "ankle_right",
            "hip_left",
            "knee_left",
            "ankle_left",
            "shoulder_middle",
            "nose",
            "shoulder_right",
            "elbow_right",
            "wrist_right",
            "shoulder_left",
            "elbow_left",
            "wrist_left",
        ]
    elif dim == 39:
        joint_names = [
            "hip_right",
            "hip_left",
            "knee_right",
            "knee_left",
            "ankle_right",
            "ankle_left",
            "nose",
            "shoulder_right",
            "shoulder_left",
            "elbow_right",
            "elbow_left",
            "wrist_right",
            "wrist_left",
        ]

    utils_show.visualize_pose_trajectories(
        poses_input, poses_target, np.array([]), joint_names, {}
    )
    utils_show.show()
