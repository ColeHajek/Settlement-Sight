import sys
from pathlib import Path

import numpy as np
import pyprojroot

from src.preprocessing.subtile_esd_hw02 import restitch

root = pyprojroot.here()
sys.path.append(str(root))


def main():
    import pyprojroot

    root = pyprojroot.here()
    import matplotlib.pyplot as plt

    sys.path.append(root)
    print(f"Added {root} to path.")

    # Run this code only after you have already created the subtiles in the directory "data/processed/Train/subtiles"
    stitched_sentinel2 = restitch(
        Path(root / "data/processed/Train/subtiles"),
        "sentinel2",
        "Tile1",
        (0, 4),
        (0, 4),
    )
    plt.imshow(
        np.dstack(
            [
                stitched_sentinel2[0, 3, :, :],
                stitched_sentinel2[0, 2, :, :],
                stitched_sentinel2[0, 1, :, :],
            ]
        )
    )
    plt.show()


if __name__ == "__main__":
    main()
