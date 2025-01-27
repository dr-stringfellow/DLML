import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class DarkMatterDataset(Dataset):
    def __init__(self, file_list):
        """
        Args:
            file_list: List of paths to the HDF5 files
        """
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_list[idx], 'r') as h5file:
            # Load data
            traces_ER = h5file['traces_ER'][:]  # Shape: (50, time_steps)
            traces_NR = h5file['traces_NR'][:]  # Shape: (50, time_steps)
            energies = h5file['energies'][:]
            positions = h5file['positions'][:]  # Target positions (x, y, z)
            times = h5file['times'][:]

        # Use either traces_ER or traces_NR (example assumes traces_ER)
        traces = traces_ER
        
        # Normalize data if needed (example: normalize traces to 0-1)
        traces = (traces - traces.min()) / (traces.max() - traces.min())
        
        # Return traces and positions (target for regression)
        return torch.tensor(traces, dtype=torch.float32), torch.tensor(positions, dtype=torch.float32)

