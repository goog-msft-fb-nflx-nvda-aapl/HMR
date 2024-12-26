import numpy as np
import torch

# Define input and output file paths
file_paths = {
    "3dhp": "./smpl_betas_train_3dhp_8_10.npy",
    "coco": "./smpl_betas_train_coco_474834_10.npy",
    "h36m": "./smpl_betas_train_h36m_5_10.npy",
}

mean_shape_paths = {
    "3dhp": "./smpl_betas_train_3dhp_avg.npy",
    "coco": "./smpl_betas_train_coco_avg.npy",
    "h36m": "./smpl_betas_train_h36m_avg.npy",
}

output_paths = {
    "3dhp": "./smpl_betas_train_3dhp_augmented.npy",
    "coco": "./smpl_betas_train_coco_augmented.npy",
    "h36m": "./smpl_betas_train_h36m_augmented.npy",
}

# Augmentation parameters
augment_shape = True
delta_betas_distribution = "normal"
delta_betas_std_vector = torch.tensor([1.5] * 10, device="cuda").float()  # Adjust device as needed
delta_betas_range = [-3.0, 3.0]  # For uniform distribution
smpl_augment_params = {
    "augment_shape": augment_shape,
    "delta_betas_distribution": delta_betas_distribution,
    "delta_betas_std_vector": delta_betas_std_vector,
    "delta_betas_range": delta_betas_range,
}

# Functions for sampling
def normal_sample_shape(batch_size, mean_shape, std_vector):
    device = mean_shape.device
    delta_betas = torch.randn(batch_size, 10, device=device) * std_vector
    shape = delta_betas + mean_shape
    return shape

def uniform_sample_shape(batch_size, mean_shape, delta_betas_range):
    device = mean_shape.device
    delta_betas = torch.empty(batch_size, 10, device=device).uniform_(
        delta_betas_range[0], delta_betas_range[1]
    )
    shape = delta_betas + mean_shape
    return shape

def augment_smpl(orig_shape, mean_shape, smpl_augment_params, augment=True):
    augment_shape = smpl_augment_params["augment_shape"] and augment
    delta_betas_distribution = smpl_augment_params["delta_betas_distribution"]
    delta_betas_range = smpl_augment_params["delta_betas_range"]
    delta_betas_std_vector = smpl_augment_params["delta_betas_std_vector"]
    batch_size = orig_shape.shape[0]

    assert delta_betas_distribution in ["uniform", "normal"]
    if delta_betas_distribution == "uniform":
        new_shape = uniform_sample_shape(batch_size, mean_shape, delta_betas_range)
    elif delta_betas_distribution == "normal":
        assert delta_betas_std_vector is not None
        new_shape = normal_sample_shape(batch_size, mean_shape, delta_betas_std_vector)

    return new_shape

# Process each dataset
device = "cuda"  # Change to "cpu" if GPU is not available
for key in file_paths:
    # Load original betas
    orig_betas = np.load(file_paths[key])

    # Load mean shape
    mean_shape_np = np.load(mean_shape_paths[key])
    mean_shape_tensor = torch.tensor(mean_shape_np, device=device, dtype=torch.float32)

    # Convert original betas to torch tensor
    orig_betas_tensor = torch.tensor(orig_betas, device=device, dtype=torch.float32)

    # Generate augmented betas
    augmented_betas_tensor = augment_smpl(
        orig_shape=orig_betas_tensor,
        mean_shape=mean_shape_tensor,
        smpl_augment_params=smpl_augment_params,
        augment=True,
    )

    # Convert back to numpy and save
    augmented_betas = augmented_betas_tensor.cpu().numpy()
    np.save(output_paths[key], augmented_betas)

    print(f"Augmented betas for {key} saved to {output_paths[key]}")