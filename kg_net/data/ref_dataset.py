import os
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as data

from data.transforms import totensor
from data.util import (sens_reduce, complex_to_2ch, paths_from_folder,
                       paths_from_ann_file)


class RefDataset(data.Dataset):
    """Reference-based cine MRI dataset for validation / testing.

    Unlike :class:`RefCUFEDDataset`, this dataset does **not** apply
    random cropping or augmentation.  It iterates over every frame of
    every sample deterministically.

    The dataset can be configured to use a subset of files via an
    annotation file (``ann_file``).

    Args:
        opt (dict): Dataset options. Required keys:
            ksp_dir (str):   Path to k-space directory.
            mask_dir (str):  Path to mask directory.
            sense_dir (str): Path to sensitivity-map directory.
        Optional keys:
            ann_file (str):  Path to a text file listing filenames to use.
                             If not given, all ``.npy`` files in *ksp_dir*
                             are used.
    """

    def __init__(self, opt):
        super(RefDataset, self).__init__()
        self.opt = opt

        self.ksp_dir = opt['ksp_dir']
        self.mask_dir = opt['mask_dir']
        self.sense_dir = opt['sense_dir']

        # determine which files to use
        ann_file = opt.get('ann_file', None)
        if ann_file is not None:
            self.filenames = paths_from_ann_file(ann_file)
        else:
            self.filenames = paths_from_folder(self.ksp_dir)

        assert len(self.filenames) > 0, \
            f'No .npy files found for dataset {opt.get("name", "")}'

        # apply train/val split if configured
        split_ratio = opt.get('split_ratio', None)
        if split_ratio is not None:
            is_train = opt.get('is_train', False)
            self._apply_split(is_train, split_ratio)

        # build flat index of (file_idx, frame_idx)
        self.samples = []
        for file_idx, fname in enumerate(self.filenames):
            ksp_path = osp.join(self.ksp_dir, fname)
            ksp = np.load(ksp_path, mmap_mode='r')
            num_frames = ksp.shape[0]
            for t in range(num_frames):
                self.samples.append((file_idx, t, num_frames))

    def _apply_split(self, is_train, split_ratio):
        """Split filenames into train/val with a stable shuffle.

        Following the pattern from image_dataset.py: sort the filenames,
        shuffle with a fixed seed for reproducibility, then split by the
        given ratio.
        """
        filenames = sorted(self.filenames)
        random.seed(42)
        random.shuffle(filenames)
        num_train = int(len(filenames) * split_ratio)
        self.filenames = filenames[:num_train] if is_train else filenames[num_train:]

    def _get_ref_indices(self, t, num_frames):
        """Get temporal reference indices with reflection padding."""
        if num_frames == 1:
            return t, t
        t_prev = t - 1 if t > 0 else t + 1
        t_next = t + 1 if t < num_frames - 1 else t - 1
        return t_prev, t_next

    def __getitem__(self, index):
        file_idx, t, num_frames = self.samples[index]
        fname = self.filenames[file_idx]

        # load data
        ksp = np.load(osp.join(self.ksp_dir, fname))      # [T, C, H, W]
        mask = np.load(osp.join(self.mask_dir, fname))     # [T, H, W]
        sense = np.load(osp.join(self.sense_dir, fname))   # [T, C, H, W]

        # reference frame indices
        t_prev, t_next = self._get_ref_indices(t, num_frames)

        # fully-sampled ground truth
        gt = sens_reduce(ksp[t], sense[t], mask=None)

        # undersampled input
        img_in_lq = sens_reduce(ksp[t], sense[t], mask=mask[t])

        # undersampled reference frames
        img_ref1 = sens_reduce(ksp[t_prev], sense[t_prev], mask=mask[t_prev])
        img_ref2 = sens_reduce(ksp[t_next], sense[t_next], mask=mask[t_next])

        # convert to 2-channel format
        gt_2ch = complex_to_2ch(gt)
        lq_2ch = complex_to_2ch(img_in_lq)
        ref1_2ch = complex_to_2ch(img_ref1)
        ref2_2ch = complex_to_2ch(img_ref2)
        up_2ch = lq_2ch.copy()

        # mask
        dc_mask = mask[t].astype(np.float32)

        # convert to tensors (no augmentation for val/test)
        gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask = totensor(
            [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask],
            float32=True)

        # build path string for logging (filename + frame index)
        lq_path = f'{osp.splitext(fname)[0]}_frame{t:03d}'

        return {
            'img_in_lq': lq_2ch,       # [2, H, W]
            'img_ref1': ref1_2ch,       # [2, H, W]
            'img_ref2': ref2_2ch,       # [2, H, W]
            'img_ref_gt': gt_2ch,       # [2, H, W]
            'img_in_up': up_2ch,        # [2, H, W]
            'dc_mask256': dc_mask,      # [H, W]
            'lq_path2': lq_path,
        }

    def __len__(self):
        return len(self.samples)
