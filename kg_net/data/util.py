import os
import os.path as osp

import numpy as np
import torch


# --------------- MRI utilities --------------- #

def ifft2c(x):
    """Centered 2D inverse FFT (numpy).

    Args:
        x (ndarray): Complex k-space data. FFT is applied over the last
            two dimensions.

    Returns:
        ndarray: Complex image-space data.
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1)),
        axes=(-2, -1))


def fft2c(x):
    """Centered 2D FFT (numpy).

    Args:
        x (ndarray): Complex image-space data. FFT is applied over the
            last two dimensions.

    Returns:
        ndarray: Complex k-space data.
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1)),
        axes=(-2, -1))


def sens_reduce(ksp, sens, mask=None):
    """Reconstruct a coil-combined image from multi-coil k-space.

    Applies optional undersampling mask, performs IFFT, and combines
    coils using conjugate sensitivity maps.

    Args:
        ksp (ndarray): Multi-coil k-space data [C, H, W] complex.
        sens (ndarray): Coil sensitivity maps [C, H, W] complex.
        mask (ndarray | None): Undersampling mask [H, W]. If None, use
            fully-sampled k-space.

    Returns:
        ndarray: Coil-combined complex image [H, W].
    """
    if mask is not None:
        ksp = ksp * mask[None, :, :]
    img_coil = ifft2c(ksp)  # [C, H, W] complex
    return (img_coil * np.conj(sens)).sum(axis=0)  # [H, W] complex


def complex_to_2ch(x):
    """Convert a complex array to 2-channel real array.

    Args:
        x (ndarray): Complex array of shape [..., H, W].

    Returns:
        ndarray: Real array of shape [2, ..., H, W] with the real part
            in channel 0 and imaginary part in channel 1.
    """
    return np.stack([x.real, x.imag], axis=0).astype(np.float32)


def complex_abs_eval(x):
    """Compute magnitude of a 2-channel real tensor (real, imag).

    Used during evaluation to convert 2-channel network output to
    magnitude image.

    Args:
        x (Tensor): Input tensor of shape [B, 2, H, W] where channel 0
            is real and channel 1 is imaginary.

    Returns:
        Tensor: Magnitude image of shape [B, 1, H, W].
    """
    return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)


# --------------- Path generation utilities --------------- #

def paths_from_folder(ksp_dir):
    """Scan a folder and return sorted list of .npy filenames.

    Args:
        ksp_dir (str): Path to the directory containing .npy files.

    Returns:
        list[str]: Sorted list of filenames (not full paths).
    """
    filenames = sorted([
        f for f in os.listdir(ksp_dir) if f.endswith('.npy')
    ])
    return filenames


def paths_from_ann_file(ann_file):
    """Read sample filenames from an annotation file.

    Each line in the annotation file should contain one filename.

    Args:
        ann_file (str): Path to the annotation text file.

    Returns:
        list[str]: List of filenames.
    """
    filenames = []
    with open(ann_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                filenames.append(line)
    return filenames
