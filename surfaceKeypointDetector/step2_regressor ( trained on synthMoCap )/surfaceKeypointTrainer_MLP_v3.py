DATASET_FILE = "/home/iismtl519-2/Desktop/SynthMoCap/synthmocap_with_vertices256.npz"
"""
vertices_256 (95575, 6890, 3) # The 2D positions of the 6890 vertices on the (256,256) croppd and resize image in the (x,y,confidence) format.
confidence is either 1(full confidence) or 0 (no confidence)
"""

PARE_PREDS_FILE = "/home/iismtl519-2/Desktop/ScoreHMR/cache/pare/pare_feature_synthmocap.npz"
"""
pose_feats (95575, 128, 24) # pose feature extracted by PARE (Part Attention Regressor for 3D Human Body Estimation, ICCV2021)
cam_shape_feats (95575, 64, 24) # shape feature extracted by PARE.
pred_betas (95575, 10) # SMPL shape parameters predicted by PARE.
pred_pose (95575, 24, 3, 3) # SMPL pose parameters predicted by PARE.
pred_cam (95575, 3) # camera translation predicted by PARE.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime

class SurfaceKeypointDataset(Dataset):
    def __init__(self, img_features: np.ndarray, vertices_2d: np.ndarray, confidence: np.ndarray):
        """
        Args:
            img_features (np.ndarray): Combined PARE features (pose_feats + cam_shape_feats)
            vertices_2d (np.ndarray): 2D vertex positions (N, 6890, 2)
            confidence (np.ndarray): Confidence values for vertices (N, 6890)
        """
        self.img_features = img_features
        self.vertices_2d = vertices_2d
        self.confidence = confidence

    def __len__(self):
        return len(self.img_features)

    def __getitem__(self, idx):
        return {
            'features': self.img_features[idx],
            'vertices': self.vertices_2d[idx],
            'confidence': self.confidence[idx]
        }

class StandarizeImageFeatures:
    def __init__(
        self,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        self.feat_mean = torch.from_numpy(feat_mean).to(device)
        self.feat_std = torch.from_numpy(feat_std).to(device)
        
    def __call__(self, features):
        return (features - self.feat_mean) / (self.feat_std + 1e-8)

class SurfaceKeypointPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm with LayerNorm
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm with LayerNorm
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm with LayerNorm
            nn.Dropout(0.1),
            
            # Output layer: 6890 vertices * 2 coordinates
            nn.Linear(hidden_dim, 6890 * 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.network(x)
        # Reshape output to (batch_size, 6890, 2)
        return x.view(batch_size, 6890, 2)
    
def compute_masked_loss(pred_vertices: torch.Tensor, 
                       target_vertices: torch.Tensor, 
                       confidence: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 loss only for vertices with confidence > 0
    """
    # Expand confidence to match coords dimension
    confidence = confidence.unsqueeze(-1).expand(-1, -1, 2)
    
    # Compute squared L2 distance
    loss = ((pred_vertices - target_vertices) ** 2) * confidence
    
    # Average over only the valid vertices
    valid_vertices = confidence.sum()
    if valid_vertices > 0:
        loss = loss.sum() / valid_vertices
    else:
        loss = loss.sum() * 0.0  # Return 0 if no valid vertices
        
    return loss

def prepare_data(vertices_file: str, pred_file: str) -> Tuple[Dataset, Dataset]:
    """
    Prepare and split the dataset
    """
    # Load data
    vertices_data = np.load(vertices_file)
    pred_data = np.load(pred_file)
    
    # Prepare features
    pose_feats = pred_data["pose_feats"].reshape(pred_data["pose_feats"].shape[0], -1)
    cam_shape_feats = pred_data["cam_shape_feats"].reshape(pred_data['cam_shape_feats'].shape[0], -1)
    img_feats = np.concatenate((pose_feats, cam_shape_feats), axis=1)
    
    # Extract vertices and confidence
    vertices_256 = vertices_data["vertices_256"]
    vertices_2d = vertices_256[..., :2]  # Only take x, y coordinates
    confidence = vertices_256[..., 2]    # Confidence values
    
    # Calculate feature statistics for standardization
    feat_mean = img_feats.mean(axis=0)
    feat_std = img_feats.std(axis=0)
    
    # Create dataset
    dataset = SurfaceKeypointDataset(img_feats, vertices_2d, confidence)
    
    # Split dataset (90% train, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, feat_mean, feat_std

class CheckpointManager:
    def __init__(self, save_dir: str, model_name: str):
        """
        Initialize checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            model_name: Name of the model for saving
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, 
                       epoch: int,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       train_loss: float,
                       val_loss: float,
                       is_best: bool = False) -> None:
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'{self.model_name}_checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if current model is best
        if is_best:
            best_model_path = os.path.join(
                self.save_dir,
                f'{self.model_name}_best_model.pth'
            )
            torch.save(checkpoint, best_model_path)
            
            # Save best model metrics
            metrics = {
                'best_epoch': epoch,
                'best_val_loss': val_loss,
                'best_train_loss': train_loss
            }
            metrics_path = os.path.join(
                self.save_dir,
                f'{self.model_name}_best_metrics.json'
            )
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       checkpoint_path: Optional[str] = None) -> Dict:
        """
        Load model checkpoint
        """
        if checkpoint_path is None:
            # Load best model by default
            checkpoint_path = os.path.join(
                self.save_dir,
                f'{self.model_name}_best_model.pth'
            )
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                standardizer: StandarizeImageFeatures,
                device: torch.device,
                num_epochs: int = 50,
                save_dir: str = 'experiments',
                model_name: Optional[str] = None) -> None:
    """
    Training loop with checkpointing and TensorBoard logging
    """
    # Initialize model name if not provided
    if model_name is None:
        model_name = f"surface_keypoint_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory
    experiment_dir = os.path.join(save_dir, model_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(experiment_dir, model_name)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    # Save model architecture and hyperparameters
    config = {
        'model_architecture': str(model),
        'optimizer': str(optimizer),
        'scheduler': str(scheduler),
        'num_epochs': num_epochs,
        'batch_size': train_loader.batch_size,
    }
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].float().to(device)
            vertices = batch['vertices'].float().to(device)
            confidence = batch['confidence'].float().to(device)
            
            features = standardizer(features)
            
            optimizer.zero_grad()
            pred_vertices = model(features)
            loss = compute_masked_loss(pred_vertices, vertices, confidence)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].float().to(device)
                vertices = batch['vertices'].float().to(device)
                confidence = batch['confidence'].float().to(device)
                
                features = standardizer(features)
                pred_vertices = model(features)
                loss = compute_masked_loss(pred_vertices, vertices, confidence)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint and best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=is_best
        )
    
    writer.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets
    train_dataset, val_dataset, feat_mean, feat_std = prepare_data(
        DATASET_FILE,
        PARE_PREDS_FILE
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize standardizer
    standardizer = StandarizeImageFeatures(feat_mean, feat_std, device)
    
    # Create model
    input_dim = feat_mean.shape[0]
    model = SurfaceKeypointPredictor(input_dim)
    
    # Train model
    model_name = "surface_keypoint_detector_v1"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        standardizer=standardizer,
        device=device,
        num_epochs=50,
        save_dir='experiments',
        model_name=model_name
    )

if __name__ == "__main__":
    main()