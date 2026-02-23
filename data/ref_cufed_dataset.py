import glob
import os
import os.path as osp
import random

import numpy as np
import torch.utils.data as data

from data.transforms import augment, totensor
from data.util import coil_combine, undersample, normalize_rss, complex_to_2ch


class RefCUFEDDataset(data.Dataset):
    """Training dataset for reference-based cine MRI reconstruction.

    For a target frame F_t, the model takes the undersampled frame as input
    and uses the temporally adjacent frames F_{t-1} and F_{t+1} as references.

    Data directories (each containing matching ``.npy`` files per slice):
        - ksp_dir:   k-space data,          shape [T, C, H, W]  (complex)
        - mask_dir:  sampling masks,         shape [T, H, W]
        - sense_dir: coil sensitivity maps,  shape [T, C, H, W]  (complex)

    Config keys used from ``opt``:
        ksp_dir, mask_dir, sense_dir, gt_size, use_flip, use_rot
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        ksp_dir = opt['ksp_dir']
        mask_dir = opt['mask_dir']
        sense_dir = opt['sense_dir']

        self.ksp_paths = sorted(glob.glob(osp.join(ksp_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(osp.join(mask_dir, '*.npy')))
        self.sense_paths = sorted(glob.glob(osp.join(sense_dir, '*.npy')))

        assert len(self.ksp_paths) > 0, f'No .npy files found in {ksp_dir}'
        assert len(self.ksp_paths) == len(self.sense_paths), (
            f'Mismatch: {len(self.ksp_paths)} ksp files vs '
            f'{len(self.sense_paths)} sense files')

        # apply train/val split if configured
        split_ratio = opt.get('split_ratio', None)
        if split_ratio is not None:
            is_train = opt.get('is_train', True)
            self._apply_split(is_train, split_ratio)

        # Determine number of time frames from the first file
        sample_ksp = np.load(self.ksp_paths[0], mmap_mode='r')
        self.time_frames = sample_ksp.shape[0]

    def _apply_split(self, is_train, split_ratio):
        """Split file paths into train/val with a stable shuffle.

        Following the pattern from image_dataset.py: sort the paths,
        shuffle with a fixed seed for reproducibility, then split by the
        given ratio.
        """
        combined = sorted(zip(self.ksp_paths, self.mask_paths, self.sense_paths))
        random.seed(42)
        random.shuffle(combined)
        num_train = int(len(combined) * split_ratio)
        selected = combined[:num_train] if is_train else combined[num_train:]
        if selected:
            self.ksp_paths, self.mask_paths, self.sense_paths = map(list, zip(*selected))
        else:
            self.ksp_paths, self.mask_paths, self.sense_paths = [], [], []

    def __len__(self):
        return len(self.ksp_paths) * self.time_frames

    def __getitem__(self, idx):
        i = idx // self.time_frames   # slice index
        t = idx % self.time_frames    # frame index

        # --- load data ---
        ksp = np.load(self.ksp_paths[i])            # [T, C, H, W] complex
        mask = np.load(
            self.mask_paths[i % len(self.mask_paths)])  # [T, H, W]
        sense = np.load(self.sense_paths[i])         # [T, C, H, W] complex

        # --- coil combine (fully sampled) ---
        combined = coil_combine(ksp, sense)           # [T, H, W] complex

        # --- normalize ---
        _, combined = normalize_rss(combined)

        # --- temporal neighbours with replicate padding ---
        t_prev = max(t - 1, 0)
        t_next = min(t + 1, self.time_frames - 1)

        # --- undersample ---
        # Expand mask dims for broadcasting: [T, H, W] -> [T, 1, H, W] not
        # needed since undersample operates on [..., H, W] and mask is [H, W].
        img_us = undersample(combined, mask)           # [T, H, W] complex

        # --- extract frames ---
        gt = combined[t]          # [H, W] complex  (fully sampled target)
        lq = img_us[t]            # [H, W] complex  (undersampled target)
        ref1 = img_us[t_prev]     # [H, W] complex  (undersampled ref t-1)
        ref2 = img_us[t_next]     # [H, W] complex  (undersampled ref t+1)
        mask_t = mask[t]          # [H, W]

        # --- convert to 2-channel [2, H, W] ---
        gt_2ch = complex_to_2ch(gt)
        lq_2ch = complex_to_2ch(lq)
        ref1_2ch = complex_to_2ch(ref1)
        ref2_2ch = complex_to_2ch(ref2)

        # --- augmentation ---
        if self.opt.get('use_flip', False) or self.opt.get('use_rot', False):
            augmented = augment(
                [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, mask_t],
                hflip=self.opt.get('use_flip', False),
                rotation=self.opt.get('use_rot', False))
            gt_2ch, lq_2ch, ref1_2ch, ref2_2ch, mask_t = augmented

        # --- to tensor ---
        gt_2ch, lq_2ch, ref1_2ch, ref2_2ch = totensor(
            [gt_2ch, lq_2ch, ref1_2ch, ref2_2ch], float32=True)
        mask_t = totensor(mask_t, float32=True)

        # ensure mask is [1, H, W]
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        return {
            'img_in_lq': lq_2ch,        # [2, H, W]  undersampled target
            'img_ref1': ref1_2ch,        # [2, H, W]  undersampled ref (t-1)
            'img_ref2': ref2_2ch,        # [2, H, W]  undersampled ref (t+1)
            'img_ref_gt': gt_2ch,        # [2, H, W]  fully sampled GT
            'img_in_up': lq_2ch.clone(), # [2, H, W]  same as lq (scale=1)
            'dc_mask256': mask_t,        # [1, H, W]  undersampling mask
            'lq_path2': self.ksp_paths[i],
        }
