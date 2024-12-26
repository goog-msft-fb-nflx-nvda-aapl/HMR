import numpy as np
from pathlib import Path

def update_feature_stats(npy_files):
    """
    Calculate mean and std of features from multiple .npy files
    
    Args:
        npy_files (list): List of paths to .npy files
        
    Returns:
        tuple: (mean, std) of the features across all files
    """
    # Initialize list to store all features
    all_features = []
    
    # Load and concatenate all features
    for npy_file in npy_files:
        features = np.load(npy_file)
        # Ensure features are 2D (samples x features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        all_features.append(features)
    
    # Concatenate all features along first dimension
    all_features = np.concatenate(all_features, axis=0)
    
    # Calculate mean and std
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    # Verify shapes
    assert mean.shape == (512,), f"Expected mean shape (512,), got {mean.shape}"
    assert std.shape == (512,), f"Expected std shape (512,), got {std.shape}"
    
    return mean, std

def save_stats(output_path, mean, std):
    """
    Save mean and std to .npz file
    
    Args:
        output_path (str): Path to save the .npz file
        mean (np.ndarray): Mean values
        std (np.ndarray): Standard deviation values
    """
    np.savez(output_path, mean=mean, std=std)
    print(f"Saved updated statistics to {output_path}")

if __name__ == "__main__":
    npy_files = [
        "/home/iismtl519-2/Desktop/npy/embedding_h36m_train_mosh.npy",
        "/home/iismtl519-2/Desktop/npy/embedding_coco_train_eft.npy",
        "/home/iismtl519-2/Desktop/npy/embedding_mpi_inf_3dhp_train_eft_v3.npy"
    ]
    
    output_path = "/home/iismtl519-2/Desktop/ScoreHMR/data/stats/pr_512_feat_stats.npz"
    
    # Calculate new statistics
    mean, std = update_feature_stats(npy_files)
    
    # Save updated statistics
    save_stats(output_path, mean, std)
    
    # Verify the saved file
    with np.load(output_path) as data:
        print("\nVerification of saved statistics:")
        for key in data.files:
            print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")