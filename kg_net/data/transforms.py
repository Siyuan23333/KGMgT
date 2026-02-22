import random

import numpy as np
import torch


def augment(img_list, hflip=True, rotation=True, return_params=False):
    """Augment images by horizontal/vertical flip and 90-degree rotation.

    All images in the list receive the same augmentation.

    Args:
        img_list (list[ndarray]): Images in [C, H, W] format.
        hflip (bool): Whether to apply horizontal flip.
        rotation (bool): Whether to apply rotation (vertical flip + rot90).
        return_params (bool): If True, also return the augmentation params.

    Returns:
        list[ndarray]: Augmented images.
        (optional) tuple: (do_hflip, do_vflip, do_rot90) if return_params.
    """
    do_hflip = hflip and random.random() < 0.5
    do_vflip = rotation and random.random() < 0.5
    do_rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if img.ndim == 3:  # [C, H, W]
            if do_hflip:
                img = img[:, :, ::-1].copy()
            if do_vflip:
                img = img[:, ::-1, :].copy()
            if do_rot90:
                img = np.ascontiguousarray(img.transpose(0, 2, 1))
        elif img.ndim == 2:  # [H, W] (e.g., mask)
            if do_hflip:
                img = img[:, ::-1].copy()
            if do_vflip:
                img = img[::-1, :].copy()
            if do_rot90:
                img = np.ascontiguousarray(img.transpose(1, 0))
        return img

    result = [_augment(img) for img in img_list]
    if return_params:
        return result, (do_hflip, do_vflip, do_rot90)
    return result


def apply_augment(img, do_hflip, do_vflip, do_rot90):
    """Apply specific augmentation to a single image.

    Args:
        img (ndarray): Image in [C, H, W] or [H, W] format.
        do_hflip (bool): Whether to horizontally flip.
        do_vflip (bool): Whether to vertically flip.
        do_rot90 (bool): Whether to rotate 90 degrees.

    Returns:
        ndarray: Augmented image.
    """
    if img.ndim == 3:
        if do_hflip:
            img = img[:, :, ::-1].copy()
        if do_vflip:
            img = img[:, ::-1, :].copy()
        if do_rot90:
            img = np.ascontiguousarray(img.transpose(0, 2, 1))
    elif img.ndim == 2:
        if do_hflip:
            img = img[:, ::-1].copy()
        if do_vflip:
            img = img[::-1, :].copy()
        if do_rot90:
            img = np.ascontiguousarray(img.transpose(1, 0))
    return img


def totensor(imgs, float32=True):
    """Convert numpy images to torch Tensors.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        float32 (bool): Whether to convert to float32.

    Returns:
        list[Tensor] | Tensor: Converted tensors.
    """
    def _totensor(img):
        tensor = torch.from_numpy(img.copy())
        if float32:
            tensor = tensor.float()
        return tensor

    if isinstance(imgs, list):
        return [_totensor(img) for img in imgs]
    else:
        return _totensor(imgs)


def random_crop(imgs, gt_size):
    """Random crop images to gt_size.

    All images are cropped at the same location.

    Args:
        imgs (list[ndarray]): Images in [C, H, W] or [H, W] format.
        gt_size (int): Target crop size.

    Returns:
        list[ndarray]: Cropped images.
    """
    # determine spatial size from the first image
    first = imgs[0]
    if first.ndim == 3:
        _, h, w = first.shape
    else:
        h, w = first.shape

    if h <= gt_size and w <= gt_size:
        return imgs

    top = random.randint(0, max(0, h - gt_size))
    left = random.randint(0, max(0, w - gt_size))

    def _crop(img):
        if img.ndim == 3:
            return img[:, top:top + gt_size, left:left + gt_size].copy()
        else:
            return img[top:top + gt_size, left:left + gt_size].copy()

    return [_crop(img) for img in imgs]
