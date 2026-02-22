import importlib
import logging
import os
import os.path as osp

import torch
import torch.utils.data

__all__ = ['create_dataset', 'create_dataloader']

# Automatically scan and import dataset modules
# (files ending with '_dataset.py' under this directory)
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(f)[0]
    for f in sorted(os.listdir(data_folder))
    if f.endswith('_dataset.py')
]
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It contains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamically find the dataset class
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = logging.getLogger('base')
    logger.info(
        f"Dataset {dataset.__class__.__name__} - {dataset_opt['name']} "
        'is created.')
    return dataset


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            n_workers (int): Number of workers for each GPU.
            batch_size (int): Training batch size for all GPUs.
        opt (dict): Config options. Default: None.
            dist (bool): Distributed training or not.
            gpu_ids (list): GPU indexes.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
    """
    phase = dataset_opt.get('phase', 'test')
    if phase == 'train':
        if opt['dist']:  # distributed training
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:  # non-distributed training
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False)
    else:  # validation / test
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)
