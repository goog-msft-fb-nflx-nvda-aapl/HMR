import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_files = {
    'hbw_f': ('meas_hbw_f_6_5.npy', 'HBW Female', 'test'),
    'hbw_m': ('meas_hbw_m_4_5.npy', 'HBW Male', 'test'),
    '3dhp_f': ('meas_3dhp_f_4_5.npy', '3DHP Female', 'train'),
    '3dhp_m': ('meas_3dhp_m_4_5.npy', '3DHP Male', 'train'),
    'h36m_f': ('meas_h36m_f_3_5.npy', 'H36M Female', 'train'),
    'h36m_m': ('meas_h36m_m_2_5.npy', 'H36M Male', 'train'),
    'ssp3d_m': ('meas_ssp3d_m_41_5.npy', 'SSP3D Male', 'test'),
    'ssp3d_f': ('meas_ssp3d_f_21_5.npy', 'SSP3D Female', 'test')
}

# Create a list to store all data
all_data = []

# Load and process each file
for key, (file_path, label, dataset_type) in data_files.items():
    data = np.load(file_path)
    
    # Calculate BMI
    height_m = data[:, 0]  # Convert height to meters
    mass_kg = data[:, 4]
    bmi = mass_kg / (height_m ** 2)
    
    # Create a DataFrame for this file
    for i in range(data.shape[0]):
        measurements = {
            'Height (cm)': data[i, 0],
            'Chest (cm)': data[i, 1],
            'Waist (cm)': data[i, 2],
            'Hips (cm)': data[i, 3],
            'Mass (kg)': data[i, 4],
            'BMI': bmi[i],
            'Dataset': label,
            'Type': dataset_type
        }
        all_data.append(measurements)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Define custom color palettes for train, test, and 3DHP
train_palette = sns.color_palette("Blues_d")  # Lighter shades of blue for train datasets
test_palette = sns.color_palette("Reds_d")  # Lighter shades of red for HBW datasets
ssp3d_palette = sns.color_palette("YlOrBr")  # Yellow-like shades for SSP3D datasets

# Set up the figure for subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# List of measurements to plot
measurements = ['Height (cm)', 'Chest (cm)', 'Waist (cm)', 'Hips (cm)', 'Mass (kg)', 'BMI']
axes = axes.flatten()  # Flatten the 2D array of axes for easy access

# Set the style using seaborn's set_style
sns.set_style("whitegrid")

# Loop through each measurement and plot the KDE for each dataset
for idx, measurement in enumerate(measurements):
    ax = axes[idx]
    # Plotting for test datasets (HBW Female and Male, SSP3D Male 41, 21)
    sns.kdeplot(data=df[df['Dataset'] == 'HBW Female'][measurement], label='TEST:HBW Female', ax=ax, fill=True, color=test_palette[0])
    sns.kdeplot(data=df[df['Dataset'] == 'HBW Male'][measurement], label='TEST:HBW Male', ax=ax, fill=True, color=test_palette[1])
    sns.kdeplot(data=df[df['Dataset'] == 'SSP3D Female'][measurement], label='TEST:SSP3D Female', ax=ax, fill=True, color=ssp3d_palette[0])
    sns.kdeplot(data=df[df['Dataset'] == 'SSP3D Male'][measurement], label='TEST:SSP3D Male', ax=ax, fill=True, color=ssp3d_palette[1])

    # Plotting for 3DHP datasets (3DHP Female, Male)
    sns.kdeplot(data=df[df['Dataset'] == '3DHP Female'][measurement], label='TRAIN:3DHP Female', ax=ax, fill=True, color=train_palette[0])
    sns.kdeplot(data=df[df['Dataset'] == '3DHP Male'][measurement], label='TRAIN:3DHP Male', ax=ax, fill=True, color=train_palette[1])

    # Plotting for H36M datasets (H36M Female, Male)
    sns.kdeplot(data=df[df['Dataset'] == 'H36M Female'][measurement], label='TRAIN:H36M Female', ax=ax, fill=True, color=train_palette[2])
    sns.kdeplot(data=df[df['Dataset'] == 'H36M Male'][measurement], label='TRAIN:H36M Male', ax=ax, fill=True, color=train_palette[3])

    ax.set_title(f'Distribution of {measurement}', fontsize=14)
    ax.set_xlabel(f'{measurement} Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title="Dataset")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
save_path = "meas_distribution.png"
plt.savefig(save_path)
plt.close()
