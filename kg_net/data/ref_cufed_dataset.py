import os
import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data as data

from data.transforms import augment, random_crop, totensor
from data.util import sens_reduce, complex_to_2ch, paths_from_folder


class RefCUFEDDataset(data.Dataset):
    """Reference-based cine MRI dataset for training.

    For each sample, the dataset:
      1. Loads multi-coil k-space, sensitivity maps, and undersampling masks.
      2. Selects a target frame ``t`` and its temporal neighbors ``t-1``,
         ``t+1`` as references.
      3. Computes the fully-sampled ground truth via IFFT + coil combination.
      4. Computes undersampled images by masking k-space before IFFT.
      5. Optionally crops and augments (flip / rotation).

    Expected data layout (three directories with matching ``.npy`` filenames)::

        ksp_dir/
            sample001.npy   # shape [T, C, H, W] complex
        mask_dir/
            sample001.npy   # shape [T, H, W]
        sense_dir/
            sample001.npy   # shape [T, C, H, W] complex

    Args:
        opt (dict): Dataset options. Required keys:
            ksp_dir (str):   Path to k-space directory.
            mask_dir (str):  Path to mask directory.
            sense_dir (str): Path to sensitivity-map directory.
        Optional keys:
            gt_size (int):   Crop size (default: no crop).
            use_flip (bool): Enable horizontal flip augmentation.
            use_rot (bool):  Enable rotation augmentation.
    """

    def __init__(self, opt):
        super(RefCUFEDDataset, self).__init__()
        self.opt = opt

        self.ksp_dir = opt['ksp_dir']
        self.mask_dir = opt['mask_dir']
        self.sense_dir = opt['sense_dir']

        # discover all .npy sample files
        self.filenames = paths_from_folder(self.ksp_dir)
        assert len(self.filenames) > 0, \
            f'No .npy files found in {self.ksp_dir}'

        # build a flat index of (file_idx, frame_idx) pairs so that each
        # item corresponds to a single frame
        self.samples = []
        for file_idx, fname in enumerate(self.filenames):
            ksp_path = osp.join(self.ksp_dir, fname)
            # use mmap to read only the shape without loading data
            ksp = np.load(ksp_path, mmap_mode='r')
            num_frames = ksp.shape[0]
            for t in range(num_frames):
                self.samples.append((file_idx, t, num_frames))

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

        # fully-sampled ground truth (frame t)
        gt = sens_reduce(ksp[t], sense[t], mask=None)  # [H, W] complex

        # undersampled input (frame t)
        img_in_lq = sens_reduce(ksp[t], sense[t], mask=mask[t])

        # undersampled reference frames
        img_ref1 = sens_reduce(ksp[t_prev], sense[t_prev], mask=mask[t_prev])
        img_ref2 = sens_reduce(ksp[t_next], sense[t_next], mask=mask[t_next])

        # convert complex images to 2-channel (real, imag) format  [2, H, W]
        gt_2ch = complex_to_2ch(gt)
        lq_2ch = complex_to_2ch(img_in_lq)
        ref1_2ch = complex_to_2ch(img_ref1)
        ref2_2ch = complex_to_2ch(img_ref2)

        # matching image for feature extraction (same as lq since scale=1)
        up_2ch = lq_2ch.copy()

        # undersampling mask for data consistency
        dc_mask = mask[t].astype(np.float32)  # [H, W]

        # optional random crop
        gt_size = self.opt.get('gt_size', None)
        if gt_size is not None:
            imgs = random_crop(
                [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask],
                gt_size)
            gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask = imgs

        # data augmentation (flip, rotation)
        use_flip = self.opt.get('use_flip', False)
        use_rot = self.opt.get('use_rot', False)
        if use_flip or use_rot:
            imgs = augment(
                [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask],
                hflip=use_flip, rotation=use_rot)
            gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask = imgs

        # convert to tensors
        gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask = totensor(
            [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, up_2ch, dc_mask],
            float32=True)

        return {
            'img_in_lq': lq_2ch,       # [2, H, W]
            'img_ref1': ref1_2ch,       # [2, H, W]
            'img_ref2': ref2_2ch,       # [2, H, W]
            'img_ref_gt': gt_2ch,       # [2, H, W]
            'img_in_up': up_2ch,        # [2, H, W]
            'dc_mask256': dc_mask,      # [H, W]
        }

    def __len__(self):
        return len(self.samples)
