import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import numpy as np
import torch
from model import AttentionDiffusionModel, linear_noise_schedule, diffusion_loss, sample_step
import time

device = 'cuda:1'

cutoff = 100000.

def sample_old(model, condition, timesteps, data_dim):
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - linear_noise_schedule(timesteps).to(device, dtype=torch.float32), dim=0)

    
    for t in reversed(range(timesteps)):
        x = sample_step(model, x, t, condition.to(device, dtype=torch.float32), alpha_bar)
        #print(x)
    return x

def sample(model, condition, timesteps, data_dim):
    # Initialize latent variable
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)

    # Compute alpha_bar with clamping for stability
    noise_schedule = linear_noise_schedule(timesteps).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - noise_schedule, dim=0)

    # Reverse sampling loop
    for t in reversed(range(timesteps)):
        x = sample_step(model, x, t, condition.to(device, dtype=torch.float32), alpha_bar)
        #print(x)

    return x

# Define model parameters
data_dim = 4
condition_dim = 1
timesteps = 25

# Create and move the model to the appropriate device
df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)

checkpoint = torch.load('models_diffusion_big/epoch-48.pt', map_location=device)
df_model.load_state_dict(checkpoint['model'])

energies = np.geomspace(10, 1e6, 500)

for i,e in enumerate(energies):
    if i % 10 != 0:
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

    energy = np.sum(sim, axis=1).reshape(-1, 1)
    if energy[0][0] < cutoff:
        print(f"Skipping {energy[0][0]:.2f} eV")
        continue
    energy = torch.tensor(energy,device=device,dtype=torch.float32)

    #print(energy)

    #print("YYYYYYY")
    # Example usage after training
    with torch.no_grad():
        gen = sample(df_model, energy/1000000, timesteps, data_dim)
    print("XXXXXXX")
    #print(sim.shape)
    #print(energy.shape)
    #print(samples.shape)


    energy = energy.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()
    
    fig,ax = plt.subplots(figsize=(7,6))
    plt.hist(gen[:,0]*energy[0],histtype='step',bins=15,label='phonon channel',color='indianred')
    plt.hist(sim[:,0],histtype='step',bins=15,linestyle='dashed',color='indianred')
    plt.hist(gen[:,1]*energy[0],histtype='step',bins=15,label='triplet channel',color='grey')
    plt.hist(sim[:,1],histtype='step',bins=15,linestyle='dashed',color='grey')
    plt.hist(gen[:,2]*energy[0],histtype='step',bins=15,label='UV channel',color='gold')
    plt.hist(sim[:,2],histtype='step',bins=15,linestyle='dashed',color='gold')
    plt.hist(gen[:,3]*energy[0],histtype='step',bins=15,label='IR channel',color='cornflowerblue')
    plt.hist(sim[:,3],histtype='step',bins=15,linestyle='dashed',color='cornflowerblue')                                                                                                  
    plt.text(0.05,0.90,"Nuclear recoil",transform=ax.transAxes,fontsize=18)
    plt.text(0.05,0.82,"$E_\mathrm{NR}=%.0f$ eV"%energy[0],transform=ax.transAxes,fontsize=18)
    ax.set_xlabel("$E$ (eV)",labelpad=20)
    ax.set_ylabel("Arbitrary units")
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(f"/web/bmaier/public_html/delight/dm_big/gen_{i}.png",bbox_inches='tight',dpi=300)
