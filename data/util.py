import numpy as np
import torch
from scipy import fft as sfft


def coil_combine(ksp, sense_map):
    """Combine multi-coil k-space data into a single complex image.

    Steps:
        1. IFFT each coil: coil_images = ifftshift(ifft2(fftshift(ksp)))
        2. Coil combine:   combined = sum(conj(sense) * coil_images, axis=coil_dim)

    Args:
        ksp (np.ndarray): Multi-coil k-space, shape [..., C, H, W], complex.
        sense_map (np.ndarray): Sensitivity maps, shape [..., C, H, W], complex.

    Returns:
        np.ndarray: Combined complex image, shape [..., H, W].
    """
    coil_images = sfft.fftshift(
        sfft.ifft2(
            sfft.ifftshift(ksp, axes=(-2, -1)),
            axes=(-2, -1), norm='ortho'),
        axes=(-2, -1))
    # Sum over the coil dimension (second-to-last two dims are H, W)
    combined = np.sum(np.conj(sense_map) * coil_images, axis=-3)
    return combined


def undersample(image, mask):
    """Apply undersampling mask in k-space to a combined complex image.

    Steps:
        1. FFT to k-space
        2. Apply mask
        3. IFFT back to image

    Args:
        image (np.ndarray): Complex image, shape [..., H, W].
        mask (np.ndarray): Sampling mask, shape [..., H, W], float/bool.

    Returns:
        np.ndarray: Undersampled complex image, shape [..., H, W].
    """
    ksp = sfft.fftshift(
        sfft.fft2(image, axes=(-2, -1), norm='ortho'),
        axes=(-2, -1))
    ksp_masked = ksp * mask
    image_us = sfft.ifft2(
        sfft.ifftshift(ksp_masked, axes=(-2, -1)),
        axes=(-2, -1), norm='ortho')
    return image_us


def normalize_rss(images, percentile=99):
    """Normalize complex images by the p-th percentile of the RSS magnitude.

    Uses the first time frame to compute the normalization factor.

    Args:
        images (np.ndarray): Complex images, shape [T, H, W].
        percentile (int): Percentile for normalization. Default: 99.

    Returns:
        tuple: (norm_factor, normalized_images)
    """
    mag = np.abs(images[0])
    norm_factor = np.percentile(mag, percentile)
    if norm_factor < 1e-10:
        norm_factor = np.max(np.abs(images)) + 1e-10
    return norm_factor, images / norm_factor


def complex_to_2ch(image):
    """Convert complex image to 2-channel (real, imag) float32 tensor.

    Args:
        image (np.ndarray): Complex image, shape [H, W] or [..., H, W].

    Returns:
        np.ndarray: 2-channel real array, shape [2, H, W] or [2, ..., H, W].
    """
    return np.stack([image.real, image.imag], axis=0).astype(np.float32)


def complex_abs_eval(tensor):
    """Compute magnitude from a 2-channel (real, imag) tensor.

    Args:
        tensor (torch.Tensor): Shape [B, 2, H, W] or [2, H, W].

    Returns:
        torch.Tensor: Magnitude image, shape [B, 1, H, W] or [1, H, W].
    """
    if tensor.dim() == 4:
        return torch.sqrt(tensor[:, 0:1] ** 2 + tensor[:, 1:2] ** 2)
    elif tensor.dim() == 3:
        return torch.sqrt(tensor[0:1] ** 2 + tensor[1:2] ** 2)
    else:
        return torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)
