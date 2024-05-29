import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
from src.visualization.restitch_plot import restitch_and_plot
from pathlib import Path

if __name__ == '__main__':
    restitch_and_plot(options= None, datamodule= None, model = None, parent_tile_id="Tile4", satellite_type="sentinel2",rgb_bands=[3,2,1],image_dir=None)