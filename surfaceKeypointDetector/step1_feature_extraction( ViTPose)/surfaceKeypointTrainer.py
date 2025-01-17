import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
import os
from datetime import datetime


# Custom Dataset
class KeypointDataset(Dataset):
    def __init__(self, features: np.ndarray, keypoints: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.keypoints = torch.FloatTensor(keypoints)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.keypoints[idx]

# Model Architectures
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_keypoints: int, output_dim: int = 3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, num_keypoints * output_dim)
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.final_layer(features)
        return output.view(-1, self.num_keypoints, self.output_dim)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, num_keypoints: int, output_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 512)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=6
        )
        self.output_proj = nn.Linear(512, num_keypoints * output_dim)
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = self.output_proj(x.squeeze(1))
        return x.view(-1, self.num_keypoints, self.output_dim)

# Loss Functions
class GaussianNLLLoss(nn.Module):
    def __init__(self, lambda_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer('lambda_weights', lambda_weights)  # This will automatically move to the correct device

    def forward(self, pred, target):
        # pred: (batch_size, num_keypoints, 3) where last dim is (x, y, sigma)
        # target: (batch_size, num_keypoints, 2) where last dim is (x, y)
        
        mu = pred[..., :2]  # Extract mean predictions (x, y)
        sigma = torch.exp(pred[..., 2])  # Extract and transform sigma predictions
        
        squared_error = torch.sum((mu - target) ** 2, dim=-1)  # Sum over x,y dimensions
        loss = (torch.log(sigma ** 2) + squared_error / (2 * sigma ** 2))
        
        if self.lambda_weights is not None:
            loss = loss * self.lambda_weights
            
        return loss.mean()

class KeypointRegressor(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        num_keypoints: int,
        loss_type: str,
        learning_rate: float = 1e-4,
        lambda_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model selection
        if model_type == 'mlp':
            self.model = MLPRegressor(
                input_dim=input_dim,
                hidden_dims=[1024, 512, 256],
                num_keypoints=num_keypoints,
                output_dim=3
            )
        elif model_type == 'transformer':
            self.model = TransformerRegressor(
                input_dim=input_dim,
                num_keypoints=num_keypoints,
                output_dim=3
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss selection
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'gaussian_nll':
            self.loss_fn = GaussianNLLLoss(lambda_weights)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        self.loss_type = loss_type
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, keypoints = batch
        predictions = self(features)
        
        if self.loss_type == 'mse':
            loss = self.loss_fn(predictions[..., :2], keypoints)
        else:  # gaussian_nll
            loss = self.loss_fn(predictions, keypoints)
            
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate and log mean keypoint error
        if self.loss_type == 'mse':
            mean_keypoint_error = torch.mean(torch.sqrt(torch.sum((predictions[..., :2] - keypoints) ** 2, dim=-1)))
            self.log('train_keypoint_error', mean_keypoint_error, on_epoch=True, logger=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        features, keypoints = batch
        predictions = self(features)
        
        if self.loss_type == 'mse':
            loss = self.loss_fn(predictions[..., :2], keypoints)
        else:  # gaussian_nll
            loss = self.loss_fn(predictions, keypoints)
            
        # Log validation metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate and log mean keypoint error
        if self.loss_type == 'mse':
            mean_keypoint_error = torch.mean(torch.sqrt(torch.sum((predictions[..., :2] - keypoints) ** 2, dim=-1)))
            self.log('val_keypoint_error', mean_keypoint_error, on_epoch=True, logger=True)
            
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def train_model(
    features: np.ndarray,
    keypoints: np.ndarray,
    model_type: str = 'transformer',
    loss_type: str = 'gaussian_nll',
    batch_size: int = 32,
    num_epochs: int = 100,
    train_split: float = 0.8,
    learning_rate: float = 1e-4,
    lambda_weights: Optional[np.ndarray] = None,
    experiment_name: Optional[str] = None
):
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique experiment name if not provided
    if experiment_name is None:
        experiment_name = f"{model_type}_{loss_type}_{timestamp}"

    # Create output directories
    output_dir = "training_outputs"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", experiment_name)
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Create dataset
    dataset = KeypointDataset(features, keypoints)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    # Convert lambda weights to tensor if provided
    if lambda_weights is not None:
        lambda_weights = torch.FloatTensor(lambda_weights)
    
    # Initialize model
    model = KeypointRegressor(
        model_type=model_type,
        input_dim=features.shape[1],
        num_keypoints=keypoints.shape[1],
        loss_type=loss_type,
        learning_rate=learning_rate,
        lambda_weights=lambda_weights
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        save_last=True  # Save last model
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Setup tensorboard logger
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name=experiment_name,
        version=timestamp
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save best model in a separate format (not just checkpoint)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': model.hparams,
        'best_val_loss': checkpoint_callback.best_model_score.item(),
        'experiment_name': experiment_name,
        'timestamp': timestamp
    }, best_model_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print(f"Tensorboard logs saved at: {os.path.join(tensorboard_dir, experiment_name)}")
    
    return model, trainer, checkpoint_callback.best_model_path

# Example usage
if __name__ == "__main__":
    # Load data
    features = np.load("/home/iismtl519-2/Desktop/SynthMoCap/synth_body/features_vitpose.npy")
    keypoints = np.load("/home/iismtl519-2/Desktop/SynthMoCap/surfaceKP_v3_resized.npy")
    
    # Optional: Create lambda weights (example: uniform weights)
    lambda_weights = np.ones(100)  # Assuming 100 keypoints
    
    # Train model
    model, trainer, best_model_path = train_model(
        features=features,
        keypoints=keypoints,
        model_type='transformer',
        loss_type='gaussian_nll',
        batch_size=32,
        num_epochs=100,
        lambda_weights=lambda_weights,
        experiment_name='keypoint_regression_experiment'
    )

# Load best model (example)
def load_best_model(model_path: str):
    checkpoint = torch.load(model_path)
    model = KeypointRegressor(
        model_type=checkpoint['hyperparameters']['model_type'],
        input_dim=checkpoint['hyperparameters']['input_dim'],
        num_keypoints=checkpoint['hyperparameters']['num_keypoints'],
        loss_type=checkpoint['hyperparameters']['loss_type'],
        learning_rate=checkpoint['hyperparameters']['learning_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model