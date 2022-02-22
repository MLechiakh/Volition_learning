import torch
import numpy as np



"""
Utility functions used in "volition_fake_data.py"

Main file is "ml_train.py"
"""


def rescale_rating(rating):
    """rescales from [0,100] to [-1,1] float"""
    return rating / 50 - 1


def _unscale_rating(r):
    """Converts [-1,1] to [0, 100]"""
    return [(index + 1) * 50 for index in r]


def scale_weights(weight, min_weight=-2, max_weight=2):
    """
    rescale from [min_weight, max_weight] to [-1,1]
    """
    return weight / ((max_weight - min_weight) / 2)

def get_crit_index(arr, device="cpu"):
    """Returns list of critere index

    arr (2D float array): one line is [userID, vID1, vID2, y_data, crit_index, rating, weight]
    device (str): device used (cpu/gpu)

    Returns:
        (float tensor): batch of ratings
    """
    return np.array([arr[:, 4]])

def get_all_vids(arr):
    """get all unique vIDs for the given user

    arr (2D float array): 1 line is [[userID, vID1, vID2, y_data, crit_index, rating, weight]

    Returns:
        (float array): unique video IDs
    """
    return np.unique(arr[:, 1:3])  # columns 1 and 2 are what we want


def get_mask(batch1, batch2):
    """returns boolean tensor indicating which videos the user rated

    batch1 (2D bool tensor): 1 line is a one-hot encoded video index
    batch2 (2D bool tensor): 1 line is a one-hot encoded video index

    Returns:
        (bool tensor): True for all indexes rated by the user
    """
    return torch.sum(batch1 + batch2, axis=0, dtype=bool)




def one_hot_vid(vid_vidx, vid):
    """One-hot inputs for neural network

    vid_vidx (dictionnary): dictionnary of {vID: idx}
    vid (int): video ID

    Returns:
        (1D boolean tensor): one-hot encoded video index
    """
    tens = torch.zeros(len(vid_vidx), dtype=bool)
    tens[vid_vidx[vid]] = True
    return tens


def one_hot_vids(vid_vidx, l_vid, device="cpu"):
    """One-hot inputs for neural network, list to batch

    vid_vidx (int dictionnary): dictionnary of {vID: vidx}
    vid (int list): list of vID
    device (str): device used (cpu/gpu)

    Returns:
        (2D boolean tensor): one line is one-hot encoded video index
    """
    batch = torch.zeros(len(l_vid), len(vid_vidx), dtype=bool, device=device)
    for idx, vid in enumerate(l_vid):
        batch[idx][vid_vidx[vid]] = True
    return batch


def get_batch_r(arr, device="cpu"):
    """Returns batch of one user's ratings

    arr (2D float array): one line is [userID, vID1, vID2, y_data, crit_index, rating, weight]
    device (str): device used (cpu/gpu)

    Returns:
        (float tensor): batch of ratings
    """
    return torch.FloatTensor(arr[:, 5], device=device)


def get_batch_w(arr, device="cpu"):
    """Returns batch of one user's weights

    arr (2D float array): one line is [userID, vID1, vID2, y_data, crit_index, rating, weight]
    device (str): device used (cpu/gpu)

    Returns:
        (float tensor): batch of ratings
    """
    return torch.FloatTensor(arr[:, 6], device=device)

def get_batch_y(arr, device="cpu"):
    """Returns batch of one user's weights

    arr (2D float array): one line is [userID, vID1, vID2, y_data, crit_index, rating, weight]
    device (str): device used (cpu/gpu)

    Returns:
        (float tensor): batch of ratings
    """
    return torch.FloatTensor(arr[:, 3], device=device)

def reverse_idxs(vids):
    """Returns dictionnary of {vid: vidx}

    vids (int iterable): unique video IDs

    Returns:
        (int:int dictionnary): dictionnary of {videoID: video index}
    """
    return {vid: idx for idx, vid in enumerate(vids)}

