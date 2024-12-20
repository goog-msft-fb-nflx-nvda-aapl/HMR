**README.md**

**H36M and 3DHP Dataset Analysis Script**

This script analyzes two human pose estimation datasets, [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), to calculate and save the average SMPL betas for each subject.

**SMPL Betas**

SMPL betas are a set of shape parameters used in the SMPL body model. They represent the body shape of an individual in the 3D model. The betas provided by this script are based on the betas provided by the author of ScoreHMR ([https://statho.github.io/ScoreHMR/](https://statho.github.io/ScoreHMR/)).

**Script Functionality**

1. **Analyzes Datasets:**
   - Loads the H36M and 3DHP datasets from provided NumPy archive (.npz) files.
   - Identifies data points belonging to each subject (S1, S2, ..., S8) based on image names.
   - Calculates the average SMPL betas for each subject.

2. **Saves Results:**
   - Saves the subject IDs and their corresponding average betas to separate NumPy archive (.npz) files for each dataset (h36m_average_betas.npz and dhp_average_betas.npz).

**Script Usage**

1. **Requirements:**
   - Python 3.x
   - NumPy library (`pip install numpy`)

2. **Execution:**
   - Place the script (`h36m_and_3dhp_analysis.py`) and the dataset files (h36m_train_mosh_v2.npz, mpi_inf_3dhp_train_eft_v3.npz) in the same directory.
   - Run the script from the command line:

   ```bash
   python h36m_and_3dhp_analysis.py
   ```

**Output**

- Two NumPy archive files (`h36m_average_betas.npz` and `dhp_average_betas.npz`) will be created in the same directory, containing the subject IDs and their average betas.

**Note:**

- This script assumes specific data structures for the H36M and 3DHP datasets provided. It may require modifications if the data structure differs.