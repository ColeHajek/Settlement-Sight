import os
import sys
import pyprojroot
root = pyprojroot.here()
from pathlib import Path

sys.path.append(str(root))
from src.preprocessing.file_utils import (
    load_satellite,
    load_satellite_dir,
)
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
)
from src.visualization.plot_utils import (
    plot_satellite_by_bands,
    #plot_viirs_by_date,
    plot_viirs_histogram,
    plot_max_projection_viirs,
    plot_gt,
    plot_gt_histogram,
    plot_sentinel2_histogram,
    plot_landsat_histogram,
    plot_sentinel1_histogram,
    plot_viirs
)

from src.utilities import SatelliteType

if __name__ == "__main__":
    tile_num = 4
    train_dir = Path(os.path.join(root, 'data', 'raw', 'Train'))

    tile_dir = train_dir / f'Tile{tile_num}'
    satellite_type = SatelliteType.GROUND_TRUTH
    

    save_plots_dir = os.path.join(root, 'visualize')
    
    #ensure directory for saving plots exists
    plt_path = Path(save_plots_dir)
    if not plt_path.exists():
        plt_path.mkdir(parents=True,exist_ok = True)
    
    test_max_viirs = False
    testHistograms = True
    testS1 = False
    testLandsat = False
    testS2 = False
    testGT = False
    test_viirs = False

    if testHistograms:
        plot_functions = {
            SatelliteType.SENTINEL_1: plot_sentinel1_histogram,
            SatelliteType.GROUND_TRUTH: plot_gt_histogram,
            SatelliteType.VIIRS: plot_viirs_histogram,
            
            SatelliteType.SENTINEL_2: plot_sentinel2_histogram,
            SatelliteType.LANDSAT: plot_landsat_histogram,
        }

        # Iterate through the SatelliteType keys and plot functions
        for satellite_type, plot_function in plot_functions.items():
            # Load the dataset stack for the current satellite type
            satellite_stack = load_satellite_dir(train_dir, satellite_type)
            
            # Call the corresponding plot function
            plot_function(satellite_stack, image_dir=save_plots_dir)
            
            # Print the status to indicate progress
            print(f"{satellite_type.name.lower()} hist done")
        

    if test_max_viirs:
        viirs_stack = load_satellite(tile_dir, "viirs")
        max_proj_viirs = maxprojection_viirs(viirs_stack)#.squeeze(0)
    
        plot_max_projection_viirs(max_proj_viirs, "VIIRS Max Projection", image_dir=save_plots_dir)
        print("plot_viirs done")
    if test_viirs:
        viirs_stack = load_satellite(tile_dir, "viirs",image_dir=save_plots_dir)
        plot_viirs(viirs_stack,'viirs',)

    if testS1:
        sentinel1_stack = load_satellite(tile_dir, "sentinel1")
        plot_satellite_by_bands( sentinel1_stack, [['VV', 'VH', 'VV-VH']], "sentinel1", image_dir=save_plots_dir)
        print("plot_s1 done")
    
    if testS2:
        sentinel2_stack = load_satellite(tile_dir, "sentinel2")
        plot_satellite_by_bands(sentinel2_stack, [['04', '03', '02'],['01'],['02'],['03'],['04'],['05'],['06'],['07'],['08'],['09'],['11'],['12'],['8A']], "sentinel2", image_dir=save_plots_dir)
        print("plot s2 done")

    if testLandsat:
        landsat_stack, file_names = load_satellite(tile_dir, "landsat")
             
        plot_satellite_by_bands(landsat_stack, file_names, [['4', '3', '2'], ['1'],['2'],['3'],['4'],['5'],['6'],['7'],['9'],['10'],['11']], "landsat", image_dir=save_plots_dir)
        print("plot landsat done")
        
    if testGT:    
        gt_stack, gt_metadata = load_satellite(tile_dir, "gt")
        plot_gt(gt_stack, "Ground Truth", image_dir=save_plots_dir)
        print("plot gt done")
    
    exit(0)