import os
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Concatenate all .npy embeddings in the specified directory.")
parser.add_argument("--in_dir", type=str, required=True, help="The path to the directory containing .npy files.")
parser.add_argument("--out_dir", type=str, required=True, help="The path to the directory for saving the integrated npy file.")
args = parser.parse_args()

# Get the directory from command-line arguments
current_directory = args.in_dir

# Validate the directory path
if not os.path.isdir(current_directory):
    raise ValueError(f"The specified directory does not exist: {current_directory}")

# Extract the directory name (e.g., "coco_train_eft_v2")
output_filename = os.path.basename(current_directory.rstrip("/"))

# Initialize a list to store arrays
all_embeddings = []

# Loop through files in the directory
for filename in sorted(os.listdir(current_directory)):
    if filename.endswith(".npy") and filename.startswith("embeddings_epoch"):
        file_path = os.path.join(current_directory, filename)
        # Load the .npy file and append to the list
        embeddings = np.load(file_path)
        all_embeddings.append(embeddings)

# Concatenate all arrays along the first axis
concatenated_embeddings = np.concatenate(all_embeddings, axis=0)

# Save the concatenated array to a new .npy file
output_path = os.path.join(args.out_dir, f"embedding_{output_filename}.npy")
np.save(output_path, concatenated_embeddings)

print(f"Concatenated embeddings saved to {output_path}"