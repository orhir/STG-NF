import os
import re

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.style.use('seaborn-ticks')


def get_ab_labels(global_data_np_ab, segs_meta_ab, path_to_vid_dir='', segs_root=''):
    pose_segs_root = segs_root
    clip_list = os.listdir(pose_segs_root)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
    labels = np.ones_like(global_data_np_ab)
    for clip in tqdm(clip_list):
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
        if type == "normal":
            continue
        clip_id = type + "_" + clip_id
        clip_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                      (segs_meta_ab[:, 0] == scene_id))[0]
        clip_metadata = segs_meta_ab[clip_metadata_inds]
        clip_res_fn = os.path.join(path_to_vid_dir, "Scene{}".format(scene_id), clip)
        filelist = sorted(os.listdir(clip_res_fn))
        clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist]
        # FIX shape bug
        clip_shapes = set([clip_gt.shape for clip_gt in clip_gt_lst])
        min_width = min([clip_shape[0] for clip_shape in clip_shapes])
        min_height = min([clip_shape[1] for clip_shape in clip_shapes])
        clip_labels = np.array([clip_gt[:min_width, :min_height] for clip_gt in clip_gt_lst])
        gt_file = os.path.join("data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
        clip_gt = np.zeros_like(clip_labels)
        with open(gt_file) as f:
            abnormality = f.readlines()
            for ab in abnormality:
                i, start, end = ab.strip("\n").split(",")
                for t in range(int(float(start)), int(float(end))):
                    clip_gt[t][clip_labels[t] == int(float(i))] = 1
        for t in range(clip_gt.shape[0]):
            if (clip_gt[t] != 0).any():  # Has abnormal event
                ab_metadata_inds = np.where(clip_metadata[:, 3].astype(int) == t)[0]
                # seg = clip_segs[ab_metadata_inds][:, :2, 0]
                clip_fig_idxs = set([arr[2] for arr in segs_meta_ab[ab_metadata_inds]])
                for person_id in clip_fig_idxs:
                    person_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                                    (segs_meta_ab[:, 0] == scene_id) &
                                                    (segs_meta_ab[:, 2] == person_id) &
                                                    (segs_meta_ab[:, 3].astype(int) == t))[0]
                    data = np.floor(global_data_np_ab[person_metadata_inds].T).astype(int)
                    if data.shape[-1] != 0:
                        if clip_gt[t][
                            np.clip(data[:, 0, 1], 0, clip_gt.shape[1] - 1),
                            np.clip(data[:, 0, 0], 0, clip_gt.shape[2] - 1)
                        ].sum() > data.shape[0] / 2:
                            # This pose is abnormal
                            labels[person_metadata_inds] = -1
    return labels[:, 0, 0, 0]


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False,
                         global_pose_data=[], dataset="ShanghaiTech"):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)
        if dataset == "UBnormal":
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = split_pose_to_segments(sing_pose_np,
                                                                                            sing_pose_meta,
                                                                                            sing_pose_keys,
                                                                                            start_ofst, seg_stride,
                                                                                            seg_len,
                                                                                            scene_id=scene_id,
                                                                                            clip_id=clip_id,
                                                                                            single_score_np=sing_scores_np,
                                                                                            dataset=dataset)
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta
    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    global_pose_data_np = np.concatenate(global_pose_data, axis=0)
    del pose_segs_data
    # del global_pose_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np
    else:
        return pose_segs_data_np, pose_segs_meta, global_pose_data_np, global_pose_data, score_segs_data_np


def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key]['scores'])
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id='', single_score_np=None, dataset="ShanghaiTech"):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    return pose_segs_np, pose_segs_meta, pose_score_np

