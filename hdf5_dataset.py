"""
PyTorch Dataset for HDF5 training data.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2


class CARLAHDF5Dataset(Dataset):
    """
    PyTorch Dataset that reads from HDF5 file created by collect_training_data.py
    """
    
    def __init__(self, h5_path, input_size=(256, 512), cache_in_memory=False):
        """
        Args:
            h5_path: Path to HDF5 file
            input_size: (height, width) to resize images to
            cache_in_memory: If True, load entire dataset into RAM (faster but memory intensive)
        """
        self.h5_path = h5_path
        self.input_size = input_size
        self.cache_in_memory = cache_in_memory
        
        # Open HDF5 file to get length
        with h5py.File(h5_path, 'r') as h5f:
            self.length = h5f['rgb'].shape[0]
            print(f"Dataset: {h5_path}")
            print(f"Total samples: {self.length}")
            
            if 'weathers' in h5f.attrs:
                print(f"Weather conditions: {list(h5f.attrs['weathers'])}")
            
            # Cache in memory if requested
            if cache_in_memory:
                print("Loading entire dataset into memory...")
                self.rgb_cache = h5f['rgb'][:]
                self.seg_cache = h5f['segmentation'][:]
                
                rgb_mem = self.rgb_cache.nbytes / (1024**3)
                seg_mem = self.seg_cache.nbytes / (1024**3)
                print(f"Memory usage: RGB={rgb_mem:.2f} GB, Labels={seg_mem:.2f} GB")
            else:
                self.rgb_cache = None
                self.seg_cache = None
                # Keep file handle open for faster access
                self.h5f = h5py.File(h5_path, 'r')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load from cache or HDF5
        if self.cache_in_memory:
            rgb = self.rgb_cache[idx].astype(np.float32)
            label = self.seg_cache[idx].astype(np.float32)
        else:
            rgb = self.h5f['rgb'][idx].astype(np.float32)
            label = self.h5f['segmentation'][idx].astype(np.float32)
        
        # Resize
        rgb = cv2.resize(rgb, (self.input_size[1], self.input_size[0]),
                        interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.input_size[1], self.input_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        
        # Normalize RGB to [0, 1]
        rgb = rgb / 255.0
        
        # Convert to torch tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        
        return rgb, label
    
    def __del__(self):
        # Close HDF5 file if it's open
        if hasattr(self, 'h5f') and self.h5f is not None:
            self.h5f.close()
