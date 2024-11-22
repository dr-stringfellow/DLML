import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import numpy as np
import torch
from model import ConditionalNormalizingFlowModel  # Importing the model from model.py
import time

if __name__ == "__main__":
    # Set up the model
    input_dim = 4
    context_dim = 1
    hidden_dim = 64
    num_layers = 5
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Instantiate the model architecture
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)

    # Load the saved model weights
    checkpoint = torch.load('models/epoch-299.pt', map_location=device)
    flow_model.load_state_dict(checkpoint['model'])

    # Switch to evaluation mode
    flow_model.eval()

    # Example: Generate samples
    # Interesting values to sample: 1.14847155e+01, 1.00462506e+02, 1.00927151e+03, 1.01393946e+04, 9.95396231e+04
    energies = np.geomspace(10, 1e6, 500)

    energies = [10., 100., 1.e3, 1.e4, 1.e5, 1.e6]

    for i,e in enumerate(energies):
        fixed_value_5th_dim = torch.tensor([[float(e)]], device=device)
        start = time.time()
        gen = flow_model.sample(num_samples=100000, context=fixed_value_5th_dim)
        end = time.time()
        print(e,end - start)

    exit(1)
    
    for i,e in enumerate(energies):
        if i % 50 != 0:
            continue

        #print(f"Loading simulated data corresponding to index {i}")

        for f in glob.glob(f"/ceph/bmaier/delight/ml/nf/data/NR_final_{i}_*.npy"):
            sim = None
            if sim is None:
                sim = np.load(f)[:, :4]
            else:
                sim = np.concatenate((sim, np.load(f)[:, :4]))                                    
            
        #print(f"Generating samples for {e} eV (index {i})")
        
        fixed_value_5th_dim = torch.tensor([[float(e)]], device=device)
        start = time.time()
        gen = flow_model.sample(num_samples=100000, context=fixed_value_5th_dim)
        end = time.time()
        gen = np.squeeze(gen.cpu().detach().numpy(), axis=0)
        #print(gen)

        
        fig,ax = plt.subplots(figsize=(7,6))
        plt.hist(gen[:,0],histtype='step',bins=15,label='phonon channel',color='indianred')
        plt.hist(sim[:,0],histtype='step',bins=15,linestyle='dashed',color='indianred')
        plt.hist(gen[:,1],histtype='step',bins=15,label='triplet channel',color='grey')
        plt.hist(sim[:,1],histtype='step',bins=15,linestyle='dashed',color='grey')
        plt.hist(gen[:,2],histtype='step',bins=15,label='UV channel',color='gold')
        plt.hist(sim[:,2],histtype='step',bins=15,linestyle='dashed',color='gold')
        #plt.hist(gen[:,3],histtype='step',bins=15,label='IR channel',color='cornflowerblue')
        #plt.hist(sim[:,3],histtype='step',bins=15,linestyle='dashed',color='cornflowerblue')
        plt.text(0.05,0.90,"Nuclear recoil",transform=ax.transAxes,fontsize=18)
        plt.text(0.05,0.82,"$E_\mathrm{NR}=%.0f$ eV"%e,transform=ax.transAxes,fontsize=18)
        ax.set_xlabel("$E$ (eV)",labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.savefig(f"/web/bmaier/public_html/delight/nf/gen_{i}.png",bbox_inches='tight',dpi=300)
        
        '''
        fig,ax = plt.subplots(figsize=(7,6))
        plt.hist(sim[:,0],histtype='step',bins=15,label='phonon channel',linestyle='dashed',color='indianred')
        plt.hist(sim[:,1],histtype='step',bins=15,label='triplet channel',linestyle='dashed',color='grey')
        plt.hist(sim[:,2],histtype='step',bins=15,label='UV channel',linestyle='dashed',color='gold')
        plt.hist(sim[:,3],histtype='step',bins=15,label='IR channel',linestyle='dashed',color='cornflowerblue')
        plt.text(0.05,0.90,"Nuclear recoil",transform=ax.transAxes,fontsize=18)
        plt.text(0.05,0.82,"$E_\mathrm{NR}=%.0f$ eV"%e,transform=ax.transAxes,fontsize=18)
        ax.set_xlabel("$E$ (eV)",labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.savefig(f"/web/bmaier/public_html/delight/sim_{i}.png",bbox_inches='tight',dpi=300)
        '''
