DATASET_FILE = "/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_vertices256.npz"
"""
vertices_256 (95575, 6890, 3) # The 2D positions of the 6890 vertices on the (256,256) croppd and resize image in the (x,y,confidence) format.
confidence is either 1(full confidence) or 0 (no confidence)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class MoCapDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        vertices = data['vertices_256']
        # Only take x,y coordinates and remove confidence
        self.vertices = torch.FloatTensor(vertices[..., :2])
        
    def __len__(self):
        return len(self.vertices)
    
    def __getitem__(self, idx):
        return self.vertices[idx]

class MoCapAutoencoder(nn.Module):
    def __init__(self, input_dim=6890*2, latent_dim=512):
        super(MoCapAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
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
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, input_dim),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(x.size(0), -1, 2)
        return reconstruction, latent
    
    def encode(self, x):
        """Method specifically for inference - returns only the embedding"""
        x = x.view(x.size(0), -1)
        return self.encoder(x)

def train_autoencoder(model, train_loader, num_epochs=100, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For saving best model
    best_loss = float('inf')
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # For plotting
    losses = []
    
    print(f"Training on {device}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Forward pass
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress within epoch
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f'autoencoder_best_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Saved best model with loss: {best_loss:.6f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_progress.png')
    plt.close()
    
    return model, save_path

def inference_on_test_data(model_path, test_data):
    """
    Apply trained encoder on test data to get embeddings
    
    Args:
        model_path (str): Path to the saved model checkpoint
        test_data (numpy.ndarray): Test data of shape (N, 6890, 2)
        
    Returns:
        numpy.ndarray: Embeddings of shape (N, 512)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MoCapAutoencoder()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Convert test data to tensor
    test_tensor = torch.FloatTensor(test_data)
    
    # Process in batches to avoid memory issues
    batch_size = 64
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i + batch_size].to(device)
            embedding = model.encode(batch)
            embeddings.append(embedding.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

# Usage example
if __name__ == "__main__":
    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 10000
    LEARNING_RATE = 1e-4
    
    dataset = MoCapDataset(DATASET_FILE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MoCapAutoencoder()
    trained_model, model_path = train_autoencoder(model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Example of inference on test data
    # Assuming you have test_data of shape (N, 6890, 2)
    # test_data = np.random.rand(1000, 6890, 2)  # Example test data
    # embeddings = inference_on_test_data(model_path, test_data)
    # print(f"Test data embeddings shape: {embeddings.shape}")  # Should be (1000, 512)