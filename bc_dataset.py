import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import hdf5plugin # Required for Zstandard compression of pixels

class ReacherHDF5Dataset(Dataset):
    def __init__(self, h5_path, mode='state', split='train', max_samples=None):
        """
        PyTorch Dataset for Behavior Cloning from Reacher HDF5.
        
        Args:
            h5_path (str): Path to the reacher.h5 dataset
            mode (str): 'state' (6D observation) or 'visual' (Pixels + TargetPos)
            split (str): 'train' (80%), 'val' (10%), or 'test' (10%)
            max_samples (int): Optional limit for fast testing
        """
        super().__init__()
        assert mode in ['state', 'visual'], "Mode must be 'state' or 'visual'"
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        
        self.mode = mode
        self.split = split
        self.h5_path = h5_path
        self.file_handle = None
        
        print(f"[{split.upper()}] Pre-computing valid indices (filtering out NaNs)...")
        with h5py.File(self.h5_path, 'r') as f:
            total_episodes = len(f['ep_len'])
            
            # Mathematical 80 / 10 / 10 Split Boundaries
            train_end = int(total_episodes * 0.8)
            val_end = int(total_episodes * 0.9)
            
            if split == 'train':
                start_ep, end_ep = 0, train_end
            elif split == 'val':
                start_ep, end_ep = train_end, val_end
            elif split == 'test':
                start_ep, end_ep = val_end, total_episodes
                
            # Get raw frame indices for these episodes
            ep_offsets = f['ep_offset'][:]
            ep_lens = f['ep_len'][:]
            
            start_frame = ep_offsets[start_ep]
            # Handle end_frame carefully if end_ep == total_episodes
            if end_ep < total_episodes:
                end_frame = ep_offsets[end_ep]
            else:
                end_frame = ep_offsets[-1] + ep_lens[-1]
                
            start_frame = int(start_frame)
            end_frame = int(end_frame)
                
            # Load only the action column for this exact split to find NaNs
            actions = f['action'][start_frame:end_frame]
            
            # Apply max_samples limit if requested (for fast iteration)
            if max_samples is not None:
                actions = actions[:max_samples]
                
            # A valid frame is one where the action is NOT NaN
            # np.isnan returns True for NaNs, so we invert it with ~
            valid_mask = ~np.isnan(actions).any(axis=1)
            
            # Get the exact index integers of all valid frames relative to the whole dataset
            self.valid_indices = np.where(valid_mask)[0] + start_frame
            
        print(f"[{split.upper()}] Dataset Initialized: Found {len(self.valid_indices)} valid frames for '{self.mode}' mode.")

    def _open_hdf5(self):
        # Open the file only when the first item is requested (worker process safe)
        if self.file_handle is None:
            self.file_handle = h5py.File(self.h5_path, 'r')
            
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        self._open_hdf5()
        
        # Get the true global index in the dataset, skipping NaNs
        global_idx = self.valid_indices[idx]
        
        # 1. Load the Target Action
        action = self.file_handle['action'][global_idx]
        action_tensor = torch.FloatTensor(action)
        
        # 2. Load the Inputs based on the Mode
        if self.mode == 'state':
            # Option A: State-Based (Just the 6-dimension physics observation)
            obs = self.file_handle['observation'][global_idx]
            obs_tensor = torch.FloatTensor(obs)
            return obs_tensor, action_tensor
            
        elif self.mode == 'visual':
            # Option B: Visual-Based (Pixels + Target Position)
            # The pixels are stored as (3, 224, 224) uint8 in [0, 255]
            pixels = self.file_handle['pixels'][global_idx]
            
            # Normalize to [0.0, 1.0] for Neural Networks
            img_tensor = torch.FloatTensor(pixels) / 255.0
            
            # Get target position
            target = self.file_handle['target_pos'][global_idx]
            target_tensor = torch.FloatTensor(target)
            
            # Return both image and target so they can be fed into the ResNet/MLP
            return img_tensor, target_tensor, action_tensor

if __name__ == "__main__":
    # Quick test to ensure splits work
    print("--- Split Tests ---")
    train_ds = ReacherHDF5Dataset('/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5', split='train', max_samples=500)
    val_ds = ReacherHDF5Dataset('/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5', split='val', max_samples=500)
    test_ds = ReacherHDF5Dataset('/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5', split='test', max_samples=500)
