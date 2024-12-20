import numpy as np

def analyze_dataset(data_path, subject_ids, key_to_analyze='betas'):
    """
    Analyzes the given dataset to:
        1. Count the number of data points for each subject.
        2. Calculate the average 'betas' for each subject.

    Args:
        data_path: Path to the .npz file containing the dataset.
        subject_ids: A list of subject IDs to analyze.
        key_to_analyze: The key in the dataset to analyze (default: 'betas').

    Returns:
        A dictionary containing:
            - 'subject_counts': A dictionary mapping subject IDs to the number of data points.
            - 'average_betas': A dictionary mapping subject IDs to the average 'betas' values.
    """

    data = np.load(data_path)
    imgnames = data['imgname']
    betas = data[key_to_analyze]

    subject_counts = {}
    average_betas = {}

    for subject_id in subject_ids:
        subject_indices = [i for i, img in enumerate(imgnames) if f"S{subject_id}" in img]
        subject_betas = betas[subject_indices]

        subject_counts[subject_id] = len(subject_indices)
        average_betas[subject_id] = np.mean(subject_betas, axis=0)

    return {'subject_counts': subject_counts, 'average_betas': average_betas}

def save_average_betas(subject_ids, average_betas, output_path):
    """
    Saves the average betas for each subject to an .npz file.

    Args:
        subject_ids: A list of subject IDs.
        average_betas: A dictionary mapping subject IDs to their average betas.
        output_path: Path to save the .npz file.
    """

    num_subjects = len(subject_ids)
    subjects = np.array(subject_ids)
    betas = np.array(list(average_betas.values()))

    np.savez(output_path, subject=subjects, beta=betas)

# Define paths and subject IDs for each dataset
PATH_TRAIN_H36M = "h36m_train_mosh_v2.npz"
H36M_SUBJECT_IDS = [1, 5, 6, 7, 8] 
H36M_OUTPUT_PATH = "smpl_betas_avg_h36m.npz"

PATH_TRAIN_3DHP = "mpi_inf_3dhp_train_eft_v3.npz"
DHP_SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
DHP_OUTPUT_PATH = "smpl_betas_avg_3dhp.npz"

# Analyze H36M dataset
h36m_results = analyze_dataset(PATH_TRAIN_H36M, H36M_SUBJECT_IDS)

# Analyze 3DHP dataset
dhp_results = analyze_dataset(PATH_TRAIN_3DHP, DHP_SUBJECT_IDS)

# Save average betas for H36M
save_average_betas(H36M_SUBJECT_IDS, h36m_results['average_betas'], H36M_OUTPUT_PATH)

# Save average betas for 3DHP
save_average_betas(DHP_SUBJECT_IDS, dhp_results['average_betas'], DHP_OUTPUT_PATH)

path_smpl_beta_3dhp = "smpl_betas_avg_3dhp.npz"
path_smpl_beta_h36m = "smpl_betas_avg_h36m.npz"
path_smpl_beta_3dhp_f = "smpl_betas_avg_3dhp_f.npy"
path_smpl_beta_3dhp_m = "smpl_betas_avg_3dhp_m.npy"
path_smpl_beta_h36m_f = "smpl_betas_avg_h36m_f.npy"
path_smpl_beta_h36m_m = "smpl_betas_avg_h36m_m.npy"

# Load data from .npz files
with np.load("smpl_betas_avg_3dhp.npz") as data_3dhp:
    subject_ids_3dhp = data_3dhp['subject']
    betas_3dhp = data_3dhp['beta']

with np.load("smpl_betas_avg_h36m.npz") as data_h36m:
    subject_ids_h36m = data_h36m['subject']
    betas_h36m = data_h36m['beta']

# Define female and male subject IDs for each dataset
female_subjects_3dhp = [1, 4, 5, 6]
male_subjects_3dhp = [2, 3, 7, 8]

female_subjects_h36m = [1, 5, 7]
male_subjects_h36m = [6, 8]

# Filter and save female and male betas for 3DHP
female_indices_3dhp = np.isin(subject_ids_3dhp, female_subjects_3dhp)
male_indices_3dhp = np.isin(subject_ids_3dhp, male_subjects_3dhp)

np.save("smpl_betas_avg_3dhp_f.npy", betas_3dhp[female_indices_3dhp])
np.save("smpl_betas_avg_3dhp_m.npy", betas_3dhp[male_indices_3dhp])

# Filter and save female and male betas for H36M
female_indices_h36m = np.isin(subject_ids_h36m, female_subjects_h36m)
male_indices_h36m = np.isin(subject_ids_h36m, male_subjects_h36m)

np.save("smpl_betas_avg_h36m_f.npy", betas_h36m[female_indices_h36m])
np.save("smpl_betas_avg_h36m_m.npy", betas_h36m[male_indices_h36m])

"""
Tranform betas into npz files taht follow bomoto input format.
"""
import numpy as np


def process_smpl_betas(file_path, output_path):
    """
    Loads average SMPL betas from a given file and creates a dictionary 
    with 'betas', 'poses', and 'trans' keys. 
    Saves the resulting dictionary to an .npz file.

    Args:
        file_path: Path to the .npy file containing average SMPL betas.
        output_path: Path to save the output .npz file.

    Returns:
        A dictionary containing:
            - 'betas': NumPy array of average betas (shape: (number_of_data, 10)), dtype: float64
            - 'poses': NumPy array of zeros (shape: (number_of_data, 72)), dtype: float64
            - 'trans': NumPy array of zeros (shape: (number_of_data, 3)), dtype: float32
    """
    betas = np.load(file_path)
    num_data = betas.shape[0]

    poses = np.zeros((num_data, 72), dtype=np.float64)
    trans = np.zeros((num_data, 3), dtype=np.float32)

    data_dict = {'betas': betas, 'poses': poses, 'trans': trans}
    np.savez(output_path, **data_dict) 

    return data_dict

# Define file paths
input_files = [
    "smpl_betas_avg_3dhp_f.npy",
    "smpl_betas_avg_3dhp_m.npy",
    "smpl_betas_avg_h36m_f.npy",
    "smpl_betas_avg_h36m_m.npy"
]

output_files = [
    "smpl_betas_avg_3dhp_f.npz",
    "smpl_betas_avg_3dhp_m.npz",
    "smpl_betas_avg_h36m_f.npz",
    "smpl_betas_avg_h36m_m.npz"
]

# Process and save each file
for input_file, output_file in zip(input_files, output_files):
    data = process_smpl_betas(input_file, output_file)