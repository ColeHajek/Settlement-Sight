import sys

import matplotlib.pyplot as plt
import numpy as np
import pyprojroot

from src.esd_data.augmentations import AddNoise, Blur, RandomHFlip, RandomVFlip
from src.esd_data.dataset import DSE

root = pyprojroot.here()
sys.path.append(str(root))
sys.path.append(root)

transforms_to_apply = [
    AddNoise(0, 0.5),
    Blur(20),
    RandomHFlip(p=1.0),
    RandomVFlip(p=1.0),
]

names = ["Noise", "Blur", "HFlip", "VFlip"]

fig, axs = plt.subplots(len(transforms_to_apply), 5)

for i, transform in enumerate(transforms_to_apply):
    dataset = DSE(
        root / "data" / "processed" / "Train" / "subtiles",
        selected_bands={"sentinel2": ["04", "03", "02"]},
        transform=transform,
    )
    X, y, metadata = dataset[0]

    X = X.reshape(4, 3, 200, 200)

    plt.suptitle(
        f"{metadata.parent_tile_id}, subtile ({metadata.x_gt}, {metadata.y_gt})"
    )

    for j in range(X.shape[0]):
        axs[i, j].set_title(f"t = {j}, tr = {names[i]}")
        axs[i, j].imshow(np.dstack([X[j, 0], X[j, 1], X[j, 2]]))
    axs[i, -1].set_title("Ground Truth")
    axs[i, -1].imshow(y[0])

plt.show()
