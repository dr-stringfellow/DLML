import os
import glob
import tqdm
import logging
import random
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import pandas as pd
import numpy as np
import torch
from model import ConditionalNormalizingFlowModel

def setup_logger():
    # Set up logging to console
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set logging level

    # Create console handler for stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


# Training the flow model
def train_conditional_flow_model(flow_model, data_train, context_train, data_val, context_val, num_epochs=1000, batch_size=512, learning_rate=1e-3):
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Convert data and context to tensors and move them to the same device as the model
    data_train = torch.tensor(data_train, dtype=torch.float32, device=flow_model.device)
    context_train = torch.tensor(context_train, dtype=torch.float32, device=flow_model.device)
    data_val = torch.tensor(data_val, dtype=torch.float32, device=flow_model.device)
    context_val = torch.tensor(context_val, dtype=torch.float32, device=flow_model.device)
    
    dataset_train = torch.utils.data.TensorDataset(data_train, context_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = torch.utils.data.TensorDataset(data_val, context_val)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Train
    all_losses_train = []
    all_losses_val = []
    for epoch in range(num_epochs):        
        total_loss_train = 0
        total_loss_val = 0
        flow_model.train()
        for batch in tqdm.tqdm(dataloader_train, desc=f"Training epoch {epoch}"):
            batch_data, batch_context = batch
            optimizer.zero_grad()
            loss = -flow_model(batch_data, batch_context).mean()  # Maximize the log probability
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Validate
        flow_model.eval()
        for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
            with torch.no_grad():
                batch_data, batch_context = batch
                loss_val = -flow_model(batch_data, batch_context).mean()
                total_loss_val += loss_val.item()
        
        # Print loss every epoch
        print(f"Epoch {epoch}, Train Loss: {total_loss_train / len(dataloader_train.dataset)}, Val Loss: {total_loss_val / len(dataloader_val.dataset)}")
        all_losses_train.append(total_loss_train / len(dataloader_train.dataset))
        all_losses_val.append(total_loss_val / len(dataloader_val.dataset))

        # Save models
        state_dicts = {'model':flow_model.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}
        torch.save(state_dicts, f'models/epoch-{epoch}.pt')
    
    # Save loss data to a CSV file
    df = pd.DataFrame({"loss_train": all_losses_train, "loss_val": all_losses_val})
    df.to_csv("loss.csv")

    fig,ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))],all_losses_train,label='train')
    plt.plot([i for i in range(len(all_losses_val))],all_losses_val,label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("/web/bmaier/public_html/delight/nf/loss.png",bbox_inches='tight',dpi=300)
    plt.savefig("/web/bmaier/public_html/delight/nf/loss.pdf",bbox_inches='tight')
    
    
def validate():
    flow_model.eval()
    for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
        with torch.no_grad():
            batch_data, batch_context = batch
            loss = -flow_model(batch_data, batch_context).mean()


def concat_files(filelist,cutoff):
    all_data = None
    for i,f in tqdm.tqdm(enumerate(filelist), total=len(filelist), desc="Loading data into array"):
        #if "lin" in f:
        #    continue
        #if i > 40:
        #    break
        # Load file and retrieve all four channels
        data = np.load(f)[:, :4]
        
        # Calculate energy as the sum of all channels
        energy = np.sum(data, axis=1).reshape(-1, 1)

        if energy[0] < cutoff:
            continue
        
        # Filter out entries below the cutoff energy
        valid_entries = energy >= cutoff
        data = data[valid_entries.ravel()]
        energy = energy[valid_entries.ravel()]

        # Concatenate data if not empty
        if all_data is None:
            all_data = np.concatenate((np.log(data), np.log(energy)), axis=1)
        else:
            all_data = np.concatenate((all_data, np.concatenate((np.log(data), np.log(energy)), axis=1)), axis=0)

    idx = [i for i in range(len(all_data))]
    print("Number of events:", len(idx))
    all_data = all_data[idx][:50000]
    
    return all_data



# Example usage
if __name__ == "__main__":
    logger = setup_logger()

    # Loading data
    cutoff_e = 0. # eV. Ingnore interactions below that.
    logger.info(f'Load data for evens with energy larger than {cutoff_e} eV.')
    files_train = glob.glob("/ceph/bmaier/delight/ml/nf/data/train/*npy")
    files_val = glob.glob("/ceph/bmaier/delight/ml/nf/data/val/*npy")
    random.seed(123)
    random.shuffle(files_train)
    data_train = concat_files(files_train,cutoff_e)
    data_val = concat_files(files_val,cutoff_e)
        
    # Separate the data into the first 4 dimensions (input) and the 5th dimension (context)
    data_train_4d = data_train[:, :4]
    context_train_5d = data_train[:, 4:5]
    data_val_4d = data_val[:, :4]
    context_val_5d = data_val[:, 4:5]

    # Initialize the conditional flow model (input dimension 4, context dimension 1, hidden dimension 64, 5 layers)
    input_dim = 4
    context_dim = 1
    hidden_dim = 96
    num_layers = 5
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    
    # Create and move the model to the appropriate device
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)
    
    # Train the model
    train_conditional_flow_model(flow_model, data_train_4d, context_train_5d, data_val_4d, context_val_5d, num_epochs=301)
    logger.info(f'Done training.')
    
