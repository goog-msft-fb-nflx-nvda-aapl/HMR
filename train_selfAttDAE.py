import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import copy
import math

class SMPLVerticesDataset(Dataset):
    def __init__(self, vertices_data, add_noise=True, noise_factor=0.05):
        """
        Dataset for SMPL vertices.
        
        Args:
            vertices_data: Numpy array of shape (num_samples, num_vertices, 3)
            add_noise: Whether to add noise for denoising
            noise_factor: Factor controlling the amount of noise
        """
        # Original shape: (num_samples, num_vertices, 3)
        # Reshape to (num_samples, num_vertices * 3) for easier processing
        self.num_samples, self.num_vertices, self.dim = vertices_data.shape
        self.vertices_flat = vertices_data.reshape(self.num_samples, -1)
        
        # Standardize the data
        self.scaler = StandardScaler()
        self.vertices_normalized = self.scaler.fit_transform(self.vertices_flat)
        
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        clean_sample = self.vertices_normalized[idx]
        
        if self.add_noise:
            # Add Gaussian noise
            noise = torch.randn_like(torch.tensor(clean_sample, dtype=torch.float32)) * self.noise_factor
            noisy_sample = clean_sample + noise.numpy()
            return torch.tensor(noisy_sample, dtype=torch.float32), torch.tensor(clean_sample, dtype=torch.float32)
        else:
            return torch.tensor(clean_sample, dtype=torch.float32), torch.tensor(clean_sample, dtype=torch.float32)
    
    def inverse_transform(self, normalized_data):
        """Transform normalized data back to original scale"""
        return self.scaler.inverse_transform(normalized_data)


class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing global dependencies"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # For self-attention, query, key, and value are the same
        residual = x
        x = self.norm(x)  # Pre-normalization
        attn_output, _ = self.attention(x, x, x)
        return residual + attn_output  # Residual connection


class FeedForward(nn.Module):
    """Feed-forward network following attention layer"""
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super(FeedForward, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4  # Default expansion ratio
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.net(x)  # Residual connection


class TransformerBlock(nn.Module):
    """Full transformer block with self-attention and feed-forward network"""
    def __init__(self, dim, num_heads=8, ff_hidden_dim=None, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForward(dim, hidden_dim=ff_hidden_dim, dropout=dropout)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class Memory_Efficient_SelfAttentionDAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_vertices, vertex_dim=3, 
                 hidden_dims=None, num_heads=8, num_transformer_layers=2,
                 patch_size=16, use_dropout=False, dropout_rate=0.1):
        """
        Memory-efficient Self-Attention based Denoising Autoencoder for SMPL vertices.
        
        Args:
            input_dim: Input dimension (num_vertices * 3)
            latent_dim: Dimension of the latent space
            num_vertices: Number of vertices in the mesh
            vertex_dim: Dimension of each vertex (typically 3 for x,y,z)
            hidden_dims: List of hidden dimensions for encoder/decoder
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer blocks
            patch_size: Size of patches for attention (number of vertices per patch)
            use_dropout: Whether to use dropout layers
            dropout_rate: Dropout probability
        """
        super(Memory_Efficient_SelfAttentionDAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_vertices = num_vertices
        self.vertex_dim = vertex_dim
        
        # Enforce patch_size to reduce memory consumption
        self.patch_size = patch_size
        
        if hidden_dims is None:
            # More memory-efficient architecture
            hidden_dims = [256, 128]
        
        # Initial projection to embedding dimension (must be divisible by num_heads)
        embed_dim = hidden_dims[0]
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads
            print(f"Adjusted embedding dimension to {embed_dim} to be divisible by {num_heads} heads")
        
        # Group vertices into patches to reduce sequence length
        self.seq_length = math.ceil(num_vertices / patch_size)
        self.token_dim = vertex_dim * patch_size
        
        print(f"Using patch_size={patch_size}, resulting in seq_length={self.seq_length}, token_dim={self.token_dim}")
        
        # Encoder
        self.initial_projection = nn.Sequential(
            nn.Linear(self.token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=embed_dim * 2,  # Reduced from 4x to 2x to save memory
                dropout=dropout_rate if use_dropout else 0.0
            ) for _ in range(num_transformer_layers)
        ])
        
        # Down-projection to latent space (using a more memory-efficient approach)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim // self.seq_length)
        )
        
        # Decoder - First expand from latent to sequence
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim // self.seq_length, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Transformer decoder layers
        self.transformer_decoder = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=embed_dim * 2,  # Reduced from 4x to 2x to save memory
                dropout=dropout_rate if use_dropout else 0.0
            ) for _ in range(num_transformer_layers)
        ])
        
        # Final projection to output dimension
        self.final_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.token_dim)
        )
    
    def reshape_to_sequence(self, x):
        """Reshape flat input to sequence for transformer"""
        batch_size = x.size(0)
        
        # Pad if necessary
        total_elements = self.seq_length * self.patch_size * self.vertex_dim
        if total_elements > self.input_dim:
            padding_size = total_elements - self.input_dim
            x = torch.cat([x, torch.zeros(batch_size, padding_size, device=x.device)], dim=1)
        
        # Group vertices into patches
        # [batch, num_vertices * 3] -> [batch, num_patches, patch_dim]
        return x.view(batch_size, self.seq_length, self.token_dim)
    
    def reshape_to_flat(self, x):
        """Reshape sequence back to flat tensor"""
        batch_size = x.size(0)
        # Remove padding if necessary
        return x.reshape(batch_size, -1)[:, :self.input_dim]
    
    def encode(self, x):
        """Get latent space representation"""
        batch_size = x.size(0)
        
        # Reshape to sequence form for transformer
        x = self.reshape_to_sequence(x)  # [batch, seq_len, token_dim]
        
        # Initial embedding
        x = self.initial_projection(x)  # [batch, seq_len, embed_dim]
        
        # Apply transformer encoder blocks
        for transformer in self.transformer_encoder:
            x = transformer(x)
        
        # Project each sequence token to a part of the latent space
        z_parts = self.to_latent(x)  # [batch, seq_len, latent_dim // seq_len]
        
        # Flatten to form the complete latent vector
        z = z_parts.reshape(batch_size, -1)  # [batch, latent_dim]
        
        return z
    
    def decode(self, z):
        """Reconstruct from latent space"""
        batch_size = z.size(0)
        
        # Reshape latent to sequence of vectors
        z_parts = z.view(batch_size, self.seq_length, -1)  # [batch, seq_len, latent_dim // seq_len]
        
        # Expand from latent parts
        x = self.from_latent(z_parts)  # [batch, seq_len, embed_dim]
        
        # Apply transformer decoder blocks
        for transformer in self.transformer_decoder:
            x = transformer(x)
        
        # Project back to token dimension
        x = self.final_projection(x)  # [batch, seq_len, token_dim]
        
        # Reshape back to flat tensor
        x = self.reshape_to_flat(x)  # [batch, num_vertices * 3]
        
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def save_checkpoint(model, optimizer, history, epoch, checkpoint_dir, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        history: Training history
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint (overwrite)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # If this is the best model, save it separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        
    print(f"Checkpoint saved at epoch {epoch}")
    return checkpoint_path


def train_attention_dae(model, train_loader, val_loader=None, num_epochs=50, 
                       learning_rate=0.001, device="cuda", weight_decay=1e-5, 
                       checkpoint_dir='checkpoints', checkpoint_freq=5,
                       early_stopping=True, patience=10, min_delta=0.001,
                       gradient_clipping=1.0, warmup_epochs=5,
                       mixed_precision=True):
    """
    Train the Self-Attention DAE with early stopping and checkpoints
    
    Args:
        model: The Self-Attention DAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: Maximum number of training epochs
        learning_rate: Maximum learning rate (after warmup)
        device: Device to train on ('cuda' or 'cpu')
        weight_decay: L2 regularization strength
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: How often to save checkpoints (epochs)
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
        gradient_clipping: Max norm for gradient clipping
        warmup_epochs: Number of epochs for learning rate warmup
        mixed_precision: Whether to use mixed precision training
    
    Returns:
        trained model and training history
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mean Squared Error loss
    criterion = nn.MSELoss(reduction='mean')
    
    # Setup for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'per_vertex_error': [],
        'learning_rates': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        per_vertex_error = 0
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        for noisy_vertices, clean_vertices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Skip batches with only one sample (BatchNorm issue)
            if noisy_vertices.size(0) <= 1:
                continue
                
            noisy_vertices = noisy_vertices.to(device)
            clean_vertices = clean_vertices.to(device)
            
            # Forward pass with mixed precision
            if mixed_precision and device == "cuda":
                with torch.cuda.amp.autocast():
                    recon_vertices, _ = model(noisy_vertices)
                    loss = criterion(recon_vertices, clean_vertices)
                
                # Backpropagation with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                recon_vertices, _ = model(noisy_vertices)
                loss = criterion(recon_vertices, clean_vertices)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate per-vertex error (in normalized space)
            batch_size = clean_vertices.size(0)
            num_vertices = clean_vertices.size(1) // 3
            
            # Use detached tensor and no_grad for memory efficiency
            with torch.no_grad():
                vertex_error = torch.mean(torch.sqrt(
                    torch.sum(
                        (recon_vertices.view(batch_size, num_vertices, 3).detach() - 
                         clean_vertices.view(batch_size, num_vertices, 3))**2, 
                        dim=2
                    )
                )).item()
                per_vertex_error += vertex_error
            
            # Clear GPU memory cache periodically
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        train_loss /= len(train_loader)
        per_vertex_error /= len(train_loader)
        
        history['train_loss'].append(train_loss)
        history['per_vertex_error'].append(per_vertex_error)
        
        # Validation
        current_val_loss = None
        is_best = False
        
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for noisy_vertices, clean_vertices in val_loader:
                    # Skip batches with only one sample (BatchNorm issue)
                    if noisy_vertices.size(0) <= 1:
                        continue
                        
                    noisy_vertices = noisy_vertices.to(device)
                    clean_vertices = clean_vertices.to(device)
                    
                    recon_vertices, _ = model(noisy_vertices)
                    loss = criterion(recon_vertices, clean_vertices)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            current_val_loss = val_loss
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, " 
                  f"Val Loss: {val_loss:.6f}, Per-vertex Error: {per_vertex_error:.6f}, "
                  f"LR: {current_lr:.6f}")
            
            # Check if this is the best model so far
            if val_loss < best_val_loss - min_delta:
                is_best = True
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs (best: {best_val_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, "
                  f"Per-vertex Error: {per_vertex_error:.6f}, LR: {current_lr:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint based on frequency or if it's the best model
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1 or is_best:
            save_checkpoint(model, optimizer, history, epoch + 1, checkpoint_dir, is_best)
        
        # Early stopping check
        if early_stopping and val_loader is not None and patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best validation loss: {best_val_loss:.6f}")
            break
    
    # Load the best model if we have validation data and did early stopping
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")
    
    return model, history


def get_embeddings(model, data_loader, device="cuda", batch_size=32):
    """
    Extract latent embeddings for all samples in the dataset
    
    Args:
        model: Trained DAE model
        data_loader: DataLoader containing the dataset
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for vertices, _ in tqdm(data_loader, desc="Extracting embeddings"):
            vertices = vertices.to(device)
            _, latent = model(vertices)
            embeddings.append(latent.cpu().numpy())
            
            # Clear cache
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.vstack(embeddings)


def reconstruct_vertices(model, embeddings, device="cuda"):
    """
    Reconstruct vertices from embeddings
    
    Args:
        model: Trained DAE model
        embeddings: Latent space embeddings
        device: Device to run inference on
        
    Returns:
        numpy array of reconstructed vertices
    """
    model.eval()
    reconstructions = []
    
    # Process in batches to avoid memory issues
    batch_size = 32  # Reduced batch size for memory efficiency
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_embeddings = embeddings[i*batch_size:(i+1)*batch_size]
            batch_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
            recon = model.decode(batch_tensor)
            reconstructions.append(recon.cpu().numpy())
            
            # Clear cache
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.vstack(reconstructions)


def plot_training_history(history, output_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['per_vertex_error'], label='Per-vertex Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Per-vertex Reconstruction Error')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to {output_path}")


def main():
    # Parameters
    input_file = "/root/smplvertices3d_synthmocap_v2.npy"
    output_dir = "attention_dae_results"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    batch_size = 16  # Reduced batch size to save memory
    num_epochs = 100
    latent_dim = 4068  # As per requirements
    
    # Memory-optimized transformer parameters
    num_heads = 4  # Reduced from 8
    num_transformer_layers = 2  # Reduced from 3
    patch_size = 16  # Group vertices into patches to reduce sequence length
    
    # Early stopping parameters
    early_stopping = True
    patience = 10
    min_delta = 0.0001
    
    # Dropout parameters
    use_dropout = True
    dropout_rate = 0.1
    
    # Learning rate parameters
    learning_rate = 0.0005
    weight_decay = 1e-4
    gradient_clipping = 1.0
    warmup_epochs = 5
    
    # Checkpoint parameters
    checkpoint_freq = 5
    
    # Enable mixed precision training
    mixed_precision = True
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = np.load(input_file)
    print(f"Data loaded with shape: {data.shape}")
    
    # Create a smaller training subset if memory is very limited
    # Uncomment these lines if still facing memory issues
    # max_samples = 20000  # Limit number of samples
    # data = data[:max_samples]
    # print(f"Limited to {max_samples} samples, new shape: {data.shape}")
    
    # Create train/val split (90/10)
    num_samples = data.shape[0]
    indices = np.random.permutation(num_samples)
    train_size = int(0.9 * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    # Create datasets
    train_dataset = SMPLVerticesDataset(train_data, add_noise=True, noise_factor=0.05)
    val_dataset = SMPLVerticesDataset(val_data, add_noise=True, noise_factor=0.05)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders with drop_last=True to avoid small batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced from 4
        drop_last=True,
        pin_memory=True  # Better memory transfer for CUDA
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # Reduced from 4
        drop_last=True,
        pin_memory=True
    )
    
    # Calculate input dimension and get vertex count
    input_dim = data.shape[1] * data.shape[2]  # 6890 vertices * 3 dimensions = 20670
    num_vertices = data.shape[1]  # 6890 vertices
    vertex_dim = data.shape[2]  # 3 dimensions (x, y, z)
    
    # Create memory-efficient model
    hidden_dims = [128, 64]  # Reduced from [512, 256]
    
    model = Memory_Efficient_SelfAttentionDAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        num_vertices=num_vertices,
        vertex_dim=vertex_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        patch_size=patch_size,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate
    )
    
    print(f"Memory-Efficient Self-Attention DAE created with:")
    print(f"- Input dim: {input_dim}")
    print(f"- Latent dim: {latent_dim}")
    print(f"- Hidden dims: {hidden_dims}")
    print(f"- Num vertices: {num_vertices}")
    print(f"- Num attention heads: {num_heads}")
    print(f"- Num transformer layers: {num_transformer_layers}")
    print(f"- Patch size: {patch_size}")
    print(f"- Using dropout: {use_dropout} (rate: {dropout_rate})")
    print(f"- Mixed precision: {mixed_precision}")
    
    # Train model with early stopping and checkpoints
    model, history = train_attention_dae(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=checkpoint_freq,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        gradient_clipping=gradient_clipping,
        warmup_epochs=warmup_epochs,
        mixed_precision=mixed_precision
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'attention_dae_final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'num_vertices': num_vertices,
        'vertex_dim': vertex_dim,
        'num_heads': num_heads,
        'num_transformer_layers': num_transformer_layers,
        'patch_size': patch_size,
        'use_dropout': use_dropout,
        'dropout_rate': dropout_rate
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Example: Getting embeddings for validation data
    print("Extracting embeddings for validation data...")
    val_embeddings = get_embeddings(model, val_loader, device)
    print(f"Embeddings shape: {val_embeddings.shape}")
    
    # Example: Reconstructing vertices from embeddings
    print("Reconstructing vertices from embeddings...")
    recon_vertices_flat = reconstruct_vertices(model, val_embeddings, device)
    
    # Transform back to original scale
    recon_vertices_original = val_dataset.inverse_transform(recon_vertices_flat)
    
    # Reshape to original format (num_samples, num_vertices, 3)
    recon_vertices_reshaped = recon_vertices_original.reshape(-1, data.shape[1], data.shape[2])
    print(f"Reconstructed vertices shape: {recon_vertices_reshaped.shape}")
    
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()