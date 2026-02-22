from re import escape
from PIL import Image
import random
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import matplotlib.pyplot as plt
import os
import glob
import scipy.io as sio
from typing import Any, Dict, List, Optional, Tuple
from scipy import fft as sfft
from improved_diffusion.mri_util import normalize_image, read_data, coil_mix, undersample_fast_scipy, undersample_fast_scipy_single




def load_data(
    *, ksp_dir, mask_dir, sense_dir, batch_size, deterministic=False,split=True,is_train=True,complex_data=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not ksp_dir or not mask_dir or not sense_dir:
        raise ValueError("unspecified data directory")
    # dataset = MRDataset(data_dir, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size(),test = test,magnitude = magnitude)
    # dataset = CINEDataset(data_dir, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size(),test = test)
    dataset = CMRDataset_2D(ksp_dir=ksp_dir, mask_dir=mask_dir, sense_dir=sense_dir, split=split,is_train=is_train, complex_data=complex_data)
    print(f"the length of dataset is {len(dataset)}")
    # dataset = CineVideoDataset(data_dir, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    while True:
        yield from loader





class CMRDataset_2D(Dataset):
    """
    Dataset for cardiac MR video (time-series) reconstruction.

    Expects three directories of matching `.npy` files:
      - ksp_dir:      k-space data with shape [T, C, H, W]
      - mask_dir:     sampling masks with shape [T, H, W]
      - sense_dir:    coil sensitivity maps with shape [T, C, H, W]

    Returns for each index:
      imgs:      np.ndarray, shape:
                  - complex=False: [1, T, H, W] (magnitude), float32 in [-1, 1]
                  - complex=True:  [2, T, H, W] (real/imag), float32 scaled by imgs_max
      con_dict:  dict with keys:
                  - 'y': undersampled image, same channel layout as imgs
                  - 'dc_ksp': fully-sampled image k-space masked by 'mask' (complex)
                  - 'maps': coil sensitivity maps [T, C, H, W]
                  - 'mask': sampling mask [T, H, W]
                  - 'combined_images_us': undersampled coil images [T, C, H, W]
                  - optional 'class' if classes provided
    """

    def __init__(
        self,
        ksp_dir: str = '/data2/hanrui/data/cmr2025/training/FullySampled',
        mask_dir: str = '/data2/hanrui/data/cmr2025/training/Mask_Radial16',
        sense_dir: str = '/data2/hanrui/data/cmr2025/training/SenseMaps',
        classes: Optional[List[int]] = None,
        num_shards: int = 1,
        time_frames: int = 12,
        complex_data: bool = True,
        split: bool = True,
        is_train: bool = True,
        uih_mask: bool = False,
    ) -> None:
        super().__init__()

        # File lists
        self.local_ksp_paths: List[str] = sorted(glob.glob(os.path.join(ksp_dir, '*.npy')))
        self.local_mask_paths: List[str] = sorted(glob.glob(os.path.join(mask_dir, '*.npy')))
        self.local_sense_map_paths: List[str] = sorted(glob.glob(os.path.join(sense_dir, '*.npy')))
        self.local_classes: Optional[List[int]] = list(classes) if classes is not None else None

        # Config
        self.uih_mask: bool = uih_mask
        self.complex: bool = complex_data
        self.time_frames: int = time_frames
        self._num_shards_unused: int = num_shards  # kept for API compatibility



        # Optional train/val split with stable shuffling
        if split:
            self._apply_split(is_train)

    def __len__(self) -> int:
        return len(self.local_ksp_paths) * self.time_frames

    def _apply_split(self, is_train: bool) -> None:
        """
        Split the dataset indices into train/val with a stable shuffle and
        update local file path lists (and classes if provided) in place.
        """
        if self.local_classes is not None:
            combined = list(zip(
                self.local_ksp_paths,
                self.local_mask_paths,
                self.local_sense_map_paths,
                self.local_classes,
            ))
        else:
            combined = list(zip(
                self.local_ksp_paths,
                self.local_mask_paths,
                self.local_sense_map_paths,
            ))

        random.seed(42)
        random.shuffle(combined)
        num_train = int(len(combined) * 0.8)
        selected = combined[:num_train] if is_train else combined[num_train:]

        if selected:
            if self.local_classes is not None:
                (self.local_ksp_paths,
                 self.local_mask_paths,
                 self.local_sense_map_paths,
                 self.local_classes) = map(list, zip(*selected))
            else:
                (self.local_ksp_paths,
                 self.local_mask_paths,
                 self.local_sense_map_paths) = map(list, zip(*selected))
        else:
            # Empty selection
            self.local_ksp_paths, self.local_mask_paths, self.local_sense_map_paths = [], [], []
            if self.local_classes is not None:
                self.local_classes = []

    def _load_ksp_mask_sense(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load per-sample, per-frame k-space, mask, and sensitivity map.
        Shapes:
          - ksp:       [T, C, H, W] (complex)
          - mask:      [T, H, W]
          - sense_map: [T, C, H, W] (complex)
        """
        ksp = np.load(self.local_ksp_paths[i])
        mask = np.load(self.local_mask_paths[i % len(self.local_mask_paths)])
        sense_map = np.load(self.local_sense_map_paths[i])
        return ksp, mask, sense_map

    def _coil_combine(self, ksp: np.ndarray, sense_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coil-domain images via IFFT (with fftshift/ifftshift) and
        combine them using sensitivity maps.

        Returns:
            coil_images: np.ndarray, coil images after IFFT
            combined:    np.ndarray, combined complex image in image domain
        """
        coil_images = sfft.fftshift(
            sfft.ifft2(sfft.ifftshift(ksp, axes=(-2, -1)), axes=(-2, -1), norm='ortho')
        )
        combined = coil_mix(coil_images, sense_map)
        return combined
    
    def _normalize_with_rss(self, images: np.ndarray,p99=True) -> np.ndarray:
        """
        Normalize the images with the max of the RSS (Root Sum of Squares) of the first coil image.
        """

        rss = np.sqrt(np.sum(np.abs(images[0])**2, axis=1, keepdims=True))
        if p99:
            max_rss = np.percentile(rss, 99)
        else:
            max_rss = np.max(rss)
        return max_rss, images / max_rss

    def _undersample_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Undersample the image with the mask.
        """
        ksp = sfft.fftshift(
            sfft.fft2(image, axes=(-2, -1), norm='ortho'), axes=(-2, -1)
        )
        ksp_under = ksp * mask



        image_under = sfft.ifft2(sfft.ifftshift(ksp_under, axes=(-2, -1)), axes=(-2, -1), norm='ortho')


 
        
        return ksp_under, image_under

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read a single sample and compute fully-sampled and undersampled images.
        """

        i  = idx // self.time_frames
        t = idx % self.time_frames


        # 1. Load original k-space, mask, and sensitivity maps
        ksp, mask, sense_map = self._load_ksp_mask_sense(i)

        # 2. Fully sampled coil images -> combined image(s)
        images_full = self._coil_combine(ksp, sense_map)  # [T, C, H, W], complex


        # 3. normalize with the max absolute value
        max_rss, images_full = self._normalize_with_rss(images_full)

    
        # 4. undersample the image
        ksps_under, images_under = self._undersample_image(images_full, mask)


        # 6. if not complex, use the absolute value
        if not self.complex:
            images_full = np.abs(images_full)[np.newaxis, :]  # [C,T, H, W]
            images_under = np.abs(images_under)[np.newaxis, :]   # [C,T, H, W]
        else:
            images_full = np.stack([images_full.real, images_full.imag], axis=0)
            images_under = np.stack([images_under.real, images_under.imag], axis=0)


        result: Dict[str, Any] = {
            'y': images_under[:,t].astype(np.float32), #[C, H, W]
            'dc_ksp': ksps_under[t], # [H,W] complex128
            'maps': sense_map[t], #[Coils, H, W] complex128
            'mask': mask[t], #[ H, W]
            'combined_images_us':images_under[:,t], #[C, H, W]
        }

        if self.local_classes is not None:
            result['class'] = np.int64(self.local_classes[idx])

        return images_full[:,t].astype(np.float32), result


        

class CMRDataset_complex(Dataset):
    """
    PyTorch Dataset for CMR (Cardiac MR) images with coil-combination, k-space manipulation,
    and mask-based undersampling.

    Returns in __getitem__:
        img_full:  [1, 256, 512], float32, fully-sampled, scaled to [-1, 1]
        con_dict: dict with
            'y':      [1, 256, 512], undersampled/composite, scaled to [-1, 1]
            'dc_img': [30, 256, 512], k-space, complex128 (UNSCALED)
            'mask':   [256, 512], float32 (binary mask; 1.0 = sampled)
    """
    def __init__(
        self,
        data_dir,
        mask_path='/data2/hanrui/data/cmr/demo_mini/masks/mask15.mat',
        classes=None,
        num_shards=1,
        time_frames=15,
        complex_data = True
    ):
        super().__init__()
        # Paths to image and sensitivity-map .mat files
        self.local_images = sorted(glob.glob(os.path.join(data_dir, 'images/*.mat')))
        self.local_maps   = sorted(glob.glob(os.path.join(data_dir, 'maps/*.mat')))
        self.local_classes = classes if classes is not None else None
        self.time_frames = time_frames
        # Mask loaded as [15, 256, 512] (frame, x, y)
        self.masks = sio.loadmat(mask_path)['masks']
        self.complex_data = complex_data
        # Scaling factors for linear scaling to [-1, 1]

        if self.complex_data:
            self.global_min_full,self.global_max_full = -0.7,1.4
            self.global_min_con,self.global_max_con = -1.02 , 2.252
        else:
            self.global_min_full,self.global_max_full = 0 , 1.43
            self.global_min_con,self.global_max_con = 0 , 2.28

    def __len__(self):
        """Total #samples = #cases * #frames."""
        return len(self.local_images) * self.time_frames

    def __getitem__(self, idx):
        """
        Main data retrieval function.
        Args:
            idx: Index over dataset
        Returns:
            img_full:  [2, 256, 512] float32, fully sampled, scaled to [-1, 1]
            con_dict: 
                'y': [2, 256, 512], undersampled/composite, scaled to [-1, 1]
                'dc_img': [30, 256, 512], k-space, complex128 (UNSCALED), not centered
                'maps': [30, 256, 512], sensitivity maps, complex128 (UNSCALED)
                'mask': [256, 512], float32 (binary mask; 1.0 = sampled)
        """
        i = idx // self.time_frames    # which patient/case
        t = idx % self.time_frames     # which time frame
        

        # Load coil images and sensitivity maps (ALL frames)
        img_full,img_con,map = self.get_comnbined_and_conidtion_image(i,t)


        dc_img,combined_images_us = self.get_dc_image(img_full,map,self.masks[t])
    


        con_dict = {
            'y': img_con,
            'dc_img': dc_img,  # [30, 256, 512], complex128
            'maps': map, # [30, 256, 512], complex128
            'mask': self.masks[t],         # [256, 512]
            'combined_images_us': combined_images_us # [ 256, 512], complex128
        }
        if self.local_classes is not None:
            con_dict["class"] = np.int64(self.local_classes[idx])

        return img_full, con_dict


    def get_comnbined_and_conidtion_image(self,i,t):
        '''
        return 
        combined_image_real: [2, 256, 512], float32, normalized
        img_con: [2, 256, 512], float32, normalized
        map: [30, 256, 512], complex128
        '''

        coil_images, maps = read_data(self.local_images[i], self.local_maps[i], self.time_frames) #[t, 30, 256, 512], complex128
        combined_images = coil_mix(coil_images, maps)
        combined_image = combined_images[t]
        combined_image_real = np.stack([(combined_image).real,(combined_image).imag],axis=0)
        combined_image_real = normalize_image(combined_image_real,self.global_min_full,self.global_max_full).astype(np.float32)


        coil_images_us, coil_kspacs_us = undersample_fast_scipy(coil_images, self.masks)
        combined_images_us = coil_mix(coil_images_us, maps)
        img_con = np.sum(combined_images_us, axis=0)
        img_con = np.stack([(img_con).real,(img_con).imag],axis=0)
        img_con = normalize_image(img_con,self.global_min_con,self.global_max_con).astype(np.float32)

        return combined_image_real,img_con,maps[t]

    def get_dc_image(self, combined_image_real, map,mask):
        '''
        return the dc image of the combined image and the map
        '''

        combined_image_complex = combined_image_real[0]+1j*combined_image_real[1]
        image_us, kspace_us = undersample_fast_scipy_single(combined_image_complex, mask)
        return kspace_us,image_us


        # coil_image = coil_unmix(combined_image_complex, map)

  
        # coil_image_us, coil_kspac_us = undersample_fast_scipy(coil_image, mask)
        # combined_images_us = coil_mix(coil_image_us, map)

        # return coil_kspac_us,combined_images_us


    
class CMRDataset(Dataset):
    """
    PyTorch Dataset for CMR (Cardiac MR) images with coil-combination, k-space manipulation,
    and mask-based undersampling.

    Returns in __getitem__:
        img_full:  [1, 256, 512], float32, fully-sampled, scaled to [-1, 1]
        con_dict: dict with
            'y':      [1, 256, 512], undersampled/composite, scaled to [-1, 1]
            'dc_img': [30, 256, 512], k-space, complex128 (UNSCALED)
            'mask':   [256, 512], float32 (binary mask; 1.0 = sampled)
    """
    def __init__(
        self,
        data_dir,
        mask_path='/data2/hanrui/data/cmr/demo_mini/masks/mask15.mat',
        classes=None,
        num_shards=1,
        time_frames=15,
    ):
        super().__init__()
        # Paths to image and sensitivity-map .mat files
        self.local_images = sorted(glob.glob(os.path.join(data_dir, 'images/*.mat')))
        self.local_maps   = sorted(glob.glob(os.path.join(data_dir, 'maps/*.mat')))
        self.local_classes = classes if classes is not None else None
        self.time_frames = time_frames
        # Mask loaded as [15, 256, 512] (frame, x, y)
        self.masks = sio.loadmat(mask_path)['masks']
        # Scaling factors for linear scaling to [-1, 1]

        self.global_min_full,self.global_max_full = 0 , 1.43
        self.global_min_con,self.global_max_con = 0 , 2.28

    def __len__(self):
        """Total #samples = #cases * #frames."""
        return len(self.local_images) * self.time_frames

    def __getitem__(self, idx):
        """
        Main data retrieval function.
        Args:
            idx: Index over dataset
        Returns:
            img_full:  [1, 256, 512] float32, fully sampled, scaled to [-1, 1]
            con_dict: 
                'y': [1, 256, 512], undersampled/composite, scaled to [-1, 1]
                'dc_img': [30, 256, 512], k-space, complex128, not centered
                'maps': [30, 256, 512], sensitivity maps, complex128 (UNSCALED)
                'mask': [256, 512], float32 (binary mask; 1.0 = sampled)
        """
        i = idx // self.time_frames    # which patient/case
        t = idx % self.time_frames     # which time frame
        

        # Load coil images and sensitivity maps (ALL frames)
        img_full,img_con,map = self.get_comnbined_and_conidtion_image(i,t)


        dc_img,combined_images_us = self.get_dc_image(img_full,map,self.masks[t])
    


        con_dict = {
            'y': img_con[None,:],
            'dc_img': dc_img,  # [30, 256, 512], complex128
            'maps': map, # [30, 256, 512], complex128
            'mask': self.masks[t],         # [256, 512]
            'combined_images_us': combined_images_us # [ 256, 512], complex128
        }
        if self.local_classes is not None:
            con_dict["class"] = np.int64(self.local_classes[idx])

        return img_full[None,:], con_dict


    def get_comnbined_and_conidtion_image(self,i,t):
        '''
        return 
        combined_image_real: [1, 256, 512], float32, normalized
        img_con: [1, 256, 512], float32, normalized
        map: [30, 256, 512], complex128
        '''

        coil_images, maps = read_data(self.local_images[i], self.local_maps[i], self.time_frames) #[t, 30, 256, 512], complex128
        combined_images = coil_mix(coil_images, maps)
        combined_image = np.abs(combined_images[t])
        combined_image = normalize_image(combined_image,self.global_min_full,self.global_max_full).astype(np.float32)


        coil_images_us, coil_kspacs_us = undersample_fast_scipy(coil_images, self.masks)
        combined_images_us = coil_mix(coil_images_us, maps)
        img_con = np.abs(np.sum(combined_images_us, axis=0))
        img_con = normalize_image(img_con,self.global_min_con,self.global_max_con).astype(np.float32)

        return combined_image,img_con,maps[t]

    def get_dc_image(self, combined_image, map,mask):
        '''
        return the dc image of the combined image and the map
        '''

        image_us, kspace_us = undersample_fast_scipy_single(combined_image, mask)
        return kspace_us,image_us





if __name__ =="__main__":

    dataset = CMRDataset_2D()
    print(len(dataset))

    import shutil
    if os.path.exists('./imgs'):
        shutil.rmtree('./imgs')
    os.makedirs('./imgs',exist_ok=True) 

    for i in range(len(dataset)):
        img_full,con_dict = dataset[i]
        img_full = np.abs(img_full[0]+1j*img_full[1])
        img_con = np.abs(con_dict['y'][0]+1j*con_dict['y'][1])

        print(img_full.max(),img_full.min(),img_con.max(),img_con.min())

        mask = con_dict['mask']

        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(img_full,cmap='gray')
        plt.title('img_full')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(img_con,cmap='gray')
        plt.title('img_con')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(mask,cmap='gray')
        plt.title('mask')
        plt.axis('off')
        plt.savefig(f'./imgs/dataset_test_{i}.png')
        plt.close()

