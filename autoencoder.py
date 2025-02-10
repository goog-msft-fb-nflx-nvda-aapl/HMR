"""
1. `MoCapDataset`: 
   - Combines pose_feats (128, 24) and cam_shape_feats (64, 24) into a single feature vector
   - Flattens the temporal dimension for processing

2. `Autoencoder`:
   - Encoder reduces input dimension to 512 through multiple linear layers
   - Uses ReLU activation and BatchNorm for better training
   - Symmetric decoder reconstructs the original features

3. Training function:
   - Implements full training loop with progress bars
   - Saves checkpoints every 10 epochs
   - Plots training loss curve
   - Uses MSE loss and Adam optimizer

4. Feature encoding function:
   - Converts all features to 512-dimensional space using trained encoder
   - Saves encoded features for downstream tasks

To use this code:

1. Make sure you have the required packages installed:
```bash
pip install torch numpy tqdm matplotlib
```

2. Run the script directly:
```bash
python your_script_name.py
```

The script will:
- Train the autoencoder
- Save checkpoints during training
- Plot the training loss
- Save the final model
- Encode all features to 512 dimensions
- Save the encoded features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class MoCapDataset(Dataset):
    def __init__(self, pose_feats, cam_shape_feats):
        # Concatenate pose_feats and cam_shape_feats along feature dimension
        # pose_feats: (N, 128, 24), cam_shape_feats: (N, 64, 24)
        self.pose_feats = torch.FloatTensor(pose_feats)
        self.cam_shape_feats = torch.FloatTensor(cam_shape_feats)
        
        # Reshape to (N, 24, 192) where 192 = 128 + 64
        self.features = torch.cat([
            self.pose_feats,
            self.cam_shape_feats
        ], dim=1)  # Concatenate along feature dimension
        
        # Flatten the temporal dimension: (N, 24, 192) -> (N, 24*192)
        self.features = self.features.reshape(self.features.shape[0], -1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

def train_autoencoder(data_path, batch_size=64, num_epochs=100, learning_rate=1e-4):
    # Load data
    data = np.load(data_path)
    pose_feats = data['pose_feats']
    cam_shape_feats = data['cam_shape_feats']
    
    # Create dataset and dataloader
    dataset = MoCapDataset(pose_feats, cam_shape_feats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Calculate input dimension
    input_dim = 24 * (128 + 64)  # 24 frames * (128 pose + 64 cam_shape) features
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in pbar:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                reconstruction, _ = model(batch)
                loss = criterion(reconstruction, batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Record average epoch loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'autoencoder_checkpoint_epoch_{epoch+1}.pt')
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), 'autoencoder_final.pt')
    return model

def encode_features(model, data_path, batch_size=64):
    """
    Use the trained encoder to convert features to latent space
    """
    # Load data
    data = np.load(data_path)
    pose_feats = data['pose_feats']
    cam_shape_feats = data['cam_shape_feats']
    
    # Create dataset and dataloader
    dataset = MoCapDataset(pose_feats, cam_shape_feats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_latent_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Encoding features'):
            batch = batch.to(device)
            _, latent = model(batch)
            all_latent_features.append(latent.cpu().numpy())
    
    # Concatenate all batches
    latent_features = np.concatenate(all_latent_features, axis=0)
    return latent_features

if __name__ == '__main__':
    data_path = '/home/iismtl519-2/Desktop/ScoreHMR/cache/pare/pare_feature_synthmocap.npz'
    
    # Train the autoencoder
    model = train_autoencoder(
        data_path,
        batch_size=64,
        num_epochs=100,
        learning_rate=1e-4
    )
    
    # Encode all features to latent space
    latent_features = encode_features(model, data_path)
    print(f'Encoded feature shape: {latent_features.shape}')
    
    # Save encoded features
    np.save('encoded_features.npy', latent_features)