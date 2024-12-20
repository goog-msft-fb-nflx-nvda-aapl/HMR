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