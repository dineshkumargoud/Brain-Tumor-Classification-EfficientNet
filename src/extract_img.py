import os
import h5py
import numpy as np
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
raw_dir = "../dataset/raw"
output_dir = "../dataset/processed"

# Create output folders
classes = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

for class_name in classes.values():
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)


# -----------------------------
# Counters
# -----------------------------
total = 0
success = 0
failed = 0


# -----------------------------
# Process all folders
# -----------------------------
for folder in sorted(os.listdir(raw_dir)):

    folder_path = os.path.join(raw_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing: {folder}")

    # Process every MAT file
    for filename in sorted(os.listdir(folder_path)):

        if not filename.endswith(".mat"):
            continue

        file_path = os.path.join(folder_path, filename)
        total += 1

        try:
            # Open MATLAB v7.3 file
            with h5py.File(file_path, "r") as f:

                cjdata = f["cjdata"]

                # Extract image
                image = np.array(cjdata["image"])

                # Extract label
                label = int(cjdata["label"][0, 0])

            # Convert image to 0-255
            image = image.astype(np.float32)

            image = (
                (image - image.min())
                / (image.max() - image.min())
                * 255
            )

            image = image.astype(np.uint8)

            # Get class name
            class_name = classes[label]

            # Remove .mat extension
            image_name = os.path.splitext(filename)[0] + ".png"

            # Output path
            output_path = os.path.join(
                output_dir,
                class_name,
                image_name
            )

            # Save PNG
            Image.fromarray(image).save(output_path)

            success += 1

        except Exception as e:

            failed += 1

            print(f"Failed: {file_path}")
            print("Error:", e)


# -----------------------------
# Final result
# -----------------------------
print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print("Total MAT files :", total)
print("Successfully saved:", success)
print("Failed           :", failed)

