import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to data files
paths = {
    "path1": "/home/iismtl519-2/Desktop/temp/HMR/visualize_BMI_distribution/smpl/3dhp_8_10/meas.npy",
    "path2": "/home/iismtl519-2/Desktop/temp/HMR/visualize_BMI_distribution/smpl/3dhp_augmented/meas.npy",
    "path3": "/home/iismtl519-2/Desktop/temp/HMR/visualize_BMI_distribution/smpl/h36m_5_10/meas.npy",
    "path4": "/home/iismtl519-2/Desktop/temp/HMR/visualize_BMI_distribution/smpl/h36m_augmented/meas.npy",
    "path5": "/home/iismtl519-2/Desktop/npy_files/measurements/meas_ssp3d_m_62_5.npy",
    "path6": "/home/iismtl519-2/Desktop/npy_files/measurements/meas_hbw_10_5.npy"
}

# Load data from paths
data = {name: np.load(path) for name, path in paths.items()}

# Define weights for training datasets
weight1 = 0.2  # Weight for path1 (3dhp_8_10)
weight3 = 0.5  # Weight for path3 (h36m_5_10)

# Define weights for augmented training datasets
weight2 = 0.2  # Weight for path2 (3dhp_augmented)
weight4 = 0.5  # Weight for path4 (h36m_augmented)

# Combine train datasets with weights
train = np.vstack([
    np.repeat(data["path1"], int(weight1 * 10), axis=0),  # Scale by weight1
    np.repeat(data["path3"], int(weight3 * 10), axis=0)   # Scale by weight3
])

# Combine train_augmented datasets with weights
train_augmented = np.vstack([
    np.repeat(data["path2"], int(weight2 * 10), axis=0),  # Scale by weight2
    np.repeat(data["path4"], int(weight4 * 10), axis=0)   # Scale by weight4
])

# Use test dataset as-is
test = data["path5"]

# Function to compute BMI
def compute_bmi(data):
    height = data[:, 0]
    mass = data[:, -1]
    return mass / height**2

# Compute BMI for each dataset
train_bmi = compute_bmi(train)
train_augmented_bmi = compute_bmi(train_augmented)
test_bmi = compute_bmi(test)

# Prepare data for visualization
columns = ["Height", "Chest", "Waist", "Hips", "Mass", "BMI"]
train_df = pd.DataFrame(np.hstack([train, train_bmi[:, None]]), columns=columns)
train_df["Dataset"] = "Train"

train_augmented_df = pd.DataFrame(np.hstack([train_augmented, train_augmented_bmi[:, None]]), columns=columns)
train_augmented_df["Dataset"] = "Train_Augmented"

test_df = pd.DataFrame(np.hstack([test, test_bmi[:, None]]), columns=columns)
test_df["Dataset"] = "Test"

combined_df = pd.concat([train_df, train_augmented_df, test_df])

# KDE Plot
plt.figure(figsize=(18, 12))
for i, metric in enumerate(columns):
    plt.subplot(2, 3, i + 1)
    sns.kdeplot(data=combined_df, x=metric, hue="Dataset", fill=True, common_norm=False, alpha=0.5)
    plt.title(f"{metric} Distribution (KDE)")
    plt.xlabel(metric)
    plt.ylabel("Density")
plt.tight_layout()
plt.savefig("kde_plots.png")

# Boxplot
plt.figure(figsize=(18, 12))
palette = {"Train": "#1f77b4", "Train_Augmented": "#ff7f0e", "Test": "#2ca02c"}  # Custom colors
for i, metric in enumerate(columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=combined_df, x="Dataset", y=metric, palette=palette)
    plt.title(f"{metric} Distribution (Boxplot)")
    plt.xlabel("Dataset")
    plt.ylabel(metric)
plt.tight_layout()
plt.savefig("box_plots.png")
