import glob
import h5py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.fft import fft
from torch.utils.data import Dataset, DataLoader

class DarkMatterDataset(Dataset):
    def __init__(self, file_list, maxevents=1e9):
        """
        Args:
            file_list: List of paths to the HDF5 files.
        """
        self.file_list = file_list
        self.data = []
        self.maxevents = maxevents

        # Load all events from all files into memory (optional: can use lazy loading for large datasets)
        for file_path in self.file_list:
            if len(self.data)> self.maxevents:
                break
            with h5py.File(file_path, 'r') as h5file:
                traces_ER = h5file['traces_ER'][:]  # Shape: (num_events, num_sensors, time_steps)
                positions = h5file['positions'][:]  # Shape: (num_events, 3)

                # Flatten event-level traces and store
                for i in range(traces_ER.shape[0]):
                    if len(self.data)> self.maxevents:
                        break
                    self.data.append((traces_ER[i], positions[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single event (traces, position)
        traces, position = self.data[idx]

        # Normalize traces (e.g., min-max or standard scaling)
        traces = (traces - traces.min()) / (traces.max() - traces.min())

        # Return as PyTorch tensors
        return torch.tensor(traces, dtype=torch.float32), torch.tensor(position, dtype=torch.float32)
    

class FourierGNN(nn.Module):
    def __init__(self, num_nodes=50, input_dim=256, hidden_dim=128, output_dim=1, num_heads=4):
        super(FourierGNN, self).__init__()
        
        # Graph Neural Network
        self.gnn1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True)
        self.gnn2 = GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=1, concat=False)
        
        # Transformer for temporal features
        self.transformer = nn.Transformer(d_model=input_dim, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, waveforms, edge_index):
        """
        Args:
            waveforms: Tensor of shape (num_nodes, time_steps)
            edge_index: Tensor of shape (2, num_edges) representing graph connections
        Returns:
            output: Tensor of shape (output_dim)
        """
        num_nodes, time_steps = waveforms.shape
        
        # Fourier Transform (amplitude and phase)
        fft_features = torch.abs(fft(waveforms))  # Amplitude
        features = torch.cat((waveforms, fft_features), dim=1)  # Combine temporal and frequency features
        
        # GNN Forward Pass
        x = self.gnn1(features, edge_index)
        x = self.relu(x)
        x = self.gnn2(x, edge_index)
        
        # Transformer for temporal sequence
        waveforms = waveforms.unsqueeze(0)  # Add batch dimension for Transformer (S, B, E)
        transformer_features = self.transformer(waveforms, waveforms)
        transformer_features = transformer_features.squeeze(0)
        
        # Fusion of GNN and Transformer features
        combined_features = torch.cat((x, transformer_features), dim=1)
        
        # Fully connected layers
        x = self.fc1(combined_features)
        x = self.relu(x)
        output = self.fc2(x)
        return output

# Example usage
num_nodes = 50
time_steps = 256
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random graph for example


file_list = glob.glob("/ceph/bmaier/delight/waveforms/v0/output_100000_*h5")
#print(file_list)

dataset = DarkMatterDataset(file_list,maxevents=300)
print("Done loading.")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


'''

# Simulate waveforms (random data)
waveforms = torch.randn(num_nodes, time_steps)




# Initialize and run the model
model = FourierGNN(num_nodes=num_nodes, input_dim=time_steps + time_steps // 2, hidden_dim=128, output_dim=1)
output = model(waveforms, edge_index)
print("Output:", output)
'''
