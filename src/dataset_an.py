import os
import h5py
from collections import Counter

# Main raw dataset folder
raw_dir = "../dataset/raw"

label_counts = Counter()
total_files = 0
failed_files = []

# Go through all folders inside raw
for folder in os.listdir(raw_dir):

    folder_path = os.path.join(raw_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    # Go through every .mat file
    for filename in os.listdir(folder_path):

        if not filename.endswith(".mat"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with h5py.File(file_path, "r") as f:

                label = int(f["cjdata"]["label"][0, 0])

                label_counts[label] += 1
                total_files += 1

        except Exception as e:
            failed_files.append((file_path, str(e)))


print("\n===== DATASET SUMMARY =====")

print("Total files:", total_files)

print("\nClass counts:")

print("Glioma:", label_counts[2])
print("Meningioma:", label_counts[1])
print("Pituitary:", label_counts[3])

print("\nFailed files:", len(failed_files))

if failed_files:
    print("\nFirst few failed files:")
    for file in failed_files[:5]:
        print(file)