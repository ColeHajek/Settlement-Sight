import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot


def main(options):

    datamodule = ESDDataModule(
        raw_dir=options.raw_dir,
        processed_dir=options.processed_dir,
        batch_size=options.batch_size,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    model = ESDSegmentation.load_from_checkpoint(options.model_path, strict=False)
    

    directory_path = Path(options.processed_dir) / 'Val' / 'subtiles'
    
    for tile in directory_path.iterdir():
        restitch_and_plot(
            options=options, 
            datamodule=datamodule, 
            model=model, 
            parent_tile_id=tile.name, 
            accelerator=options.accelerator, 
            results_dir=options.results_dir
        )

    return


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
