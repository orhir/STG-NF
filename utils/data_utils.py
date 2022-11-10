import math

import numpy as np
import torch


def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    shear_mat = torch.tensor([[1, shearx, 0], [sheary, 1, 0], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(shear_mat, aff_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, shearx, sheary, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x


trans_list = [
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True),  # 1
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0.1),  # 2
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0.1),  # 3
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0, sheary=0.1),  # 4
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0, sheary=0.1),  # 5
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0),  # 6
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0),  # 7
]


def get_global_size(pose_data, keypoint_axis=1):
    # param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, Keypoints=18, 3)
    w_list = []
    h_list = []
    cor_list = []
    for person in pose_data:
        max_kp_xy = np.max(np.abs(person[..., :2]), axis=keypoint_axis)
        min_kp_xy = np.min(np.abs(person[..., :2]), axis=keypoint_axis)
        w = max_kp_xy[..., 0] - min_kp_xy[..., 0]
        h = max_kp_xy[..., 1] - min_kp_xy[..., 1]
        cor = (max_kp_xy + min_kp_xy) / 2
        w_list.append(w)
        h_list.append(h)
        cor_list.append(cor)
    return cor_list, (w_list, h_list)


def _normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', False)
    sub_mean = kwargs.get('sub_mean', True)
    scale = kwargs.get('scale', True)
    scale_proportional = kwargs.get('scale_proportional', False)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    if sub_mean or scale or scale_proportional:  # Inner frame scaling requires mean subtraction
        pose_data_zero_mean = pose_data_centered
        mean_kp_val = np.mean(pose_data_zero_mean[..., :2], 2)
        pose_data_zero_mean[..., :2] -= mean_kp_val[:, :, None, :]

    max_kp_xy = np.max(np.abs(pose_data_centered[..., :2]), axis=2)
    max_kp_coord = max_kp_xy.max(axis=2)

    pose_data_scaled = pose_data_zero_mean
    if scale:
        # Scale sequence to maximize the [-1,1] frame
        # Removes average position from all keypoints, than scales in x and y to fill the frame
        # Loses body proportions
        pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_xy[:, None, None, :]

    elif scale_proportional:
        # Same as scale but normalizes by the same factor
        # (smaller axis, i.e. divides by larger fraction value)
        # Keeps propotions
        pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_coord[:, :, None, None]

    return pose_data_scaled


def normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', False)
    # sub_mean = kwargs.get('sub_mean', True)
    # scale = kwargs.get('scale', False)
    # scale_proportional = kwargs.get('scale_proportional', True)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    pose_data_zero_mean = pose_data_centered
    # return pose_data_zero_mean

    pose_data_zero_mean[..., :2] = (pose_data_centered[..., :2] - pose_data_centered[..., :2].mean(axis=(1, 2))[:, None, None, :]) / pose_data_centered[..., 1].std(axis=(1, 2))[:, None, None, None]
    return pose_data_zero_mean

    # if sub_mean or scale or scale_proportional:  # Inner frame scaling requires mean subtraction
    #     mean_kp_val = np.mean(pose_data_zero_mean[..., :2], (1, 2))
    #     pose_data_zero_mean[..., :2] -= mean_kp_val[:, None, None, :]
    #
    # max_kp_xy = np.max(np.abs(pose_data_centered[..., :2]), axis=(1, 2))
    # max_kp_coord = max_kp_xy.max(axis=1)
    #
    # pose_data_scaled = pose_data_zero_mean
    # if scale:
    #     # Scale sequence to maximize the [-1,1] frame
    #     # Removes average position from all keypoints, than scales in x and y to fill the frame
    #     # Loses body proportions
    #     pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_xy[:, None, None, :]
    #
    # elif scale_proportional:
    #     # Same as scale but normalizes by the same factor
    #     # (smaller axis, i.e. divides by larger fraction value)
    #     # Keeps propotions
    #     pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_coord[:, None, None, None]
    #
    # return pose_data_scaled
