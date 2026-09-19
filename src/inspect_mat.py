import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "../dataset/raw/brainTumorDataPublic_1766/1.mat"

with h5py.File(file_path, "r") as f:
    cjdata = f["cjdata"]

    # Extract image
    image = np.array(cjdata["image"])

    # Extract label
    label = np.array(cjdata["label"])

    print("Image shape:", image.shape)
    print("Label:", label)

    # Display MRI
    plt.imshow(image, cmap="gray")
    plt.title(f"Brain MRI - Label: {label}")
    plt.axis("off")
    plt.show()