import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import numpy as np
import torch
from model import AttentionDiffusionModel, linear_noise_schedule, diffusion_loss, sample_step
import time

device = 'cuda:2'

def sample(model, condition, timesteps, data_dim):
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - linear_noise_schedule(timesteps).to(device, dtype=torch.float32), dim=0)
    
    for t in reversed(range(timesteps)):
        print("ZZZZZZZZZZZ", t)
        x = sample_step(model, x, t, condition.to(device, dtype=torch.float32), alpha_bar)
        print("############")
        print(x)
    return x

# Define model parameters
data_dim = 4
condition_dim = 1
timesteps = 100

# Create and move the model to the appropriate device
df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)

checkpoint = torch.load('models_diffusion_timesteps/epoch-19.pt', map_location=device)
df_model.load_state_dict(checkpoint['model'])

energies = np.geomspace(10, 1e6, 500)

for i,e in enumerate(energies):
    if i % 50 != 0:
        continue
    print(f"Loading simulated data corresponding to index {i}")

    sim = None
    for f in glob.glob(f"/ceph/bmaier/delight/ml/nf/data/val/NR_final_{i}_*.npy"):
        if "lin" in f:
            continue
        if sim is None:
            sim = np.load(f)[:, :4]
        else:
            sim = np.concatenate((sim, np.load(f)[:, :4]))            

    
    energy = torch.tensor(np.sum(sim, axis=1).reshape(-1, 1),device=device,dtype=torch.float32)

    #print(energy)
    
    # Example usage after training
    with torch.no_grad():
        samples = sample(df_model, energy, timesteps, data_dim)
    print("XXX")
    print(sim.shape)
    print(energy.shape)
    print(samples.shape)

    print(samples)
