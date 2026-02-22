import numpy as np
import torch


def augment(img_list, hflip=True, rotation=True):
    """Apply random augmentations to a list of images (and masks).

    All items are transformed identically.  Works with arrays of shape
    [C, H, W] (images) or [H, W] (masks).

    Args:
        img_list (list[np.ndarray]): List of images / masks.
        hflip (bool): Enable horizontal flip.
        rotation (bool): Enable rotation (90-degree multiples via transpose).

    Returns:
        list[np.ndarray]: Augmented images / masks.
    """
    hflip = hflip and np.random.random() < 0.5
    vflip = rotation and np.random.random() < 0.5
    rot90 = rotation and np.random.random() < 0.5

    def _apply(img):
        if hflip:
            img = img[..., ::-1].copy()
        if vflip:
            img = img[..., ::-1, :].copy()
        if rot90:
            if img.ndim == 3:
                img = img.transpose(0, 2, 1).copy()
            else:
                img = img.transpose(1, 0).copy()
        return img

    return [_apply(x) for x in img_list]


def totensor(imgs, float32=True):
    """Convert numpy image(s) to torch Tensor(s).

    Args:
        imgs (np.ndarray | list[np.ndarray]): Input images.
        float32 (bool): Cast to float32.

    Returns:
        torch.Tensor | list[torch.Tensor]
    """

    def _to_tensor(img):
        t = torch.from_numpy(np.ascontiguousarray(img))
        if float32:
            t = t.float()
        return t

    if isinstance(imgs, list):
        return [_to_tensor(x) for x in imgs]
    return _to_tensor(imgs)
