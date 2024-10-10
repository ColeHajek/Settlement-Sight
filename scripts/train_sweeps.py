"""
train_sweeps.py runs hyperparameter search using Weights and Biases and the predefined sweep.yml file
Please adapt the sweep.yml file to your needs and run the script with the following command:
    python scripts/train_sweeps.py --sweep_file scripts/sweeps.yml

For more information on sweeps in Weights and Biases, please refer to the following link:
https://docs.wandb.ai/guides/sweeps
"""
import argparse
from pathlib import Path
import sys

import wandb
import yaml

sys.path.append(".")
from scripts.train import train
from src.utilities import ESDConfig, PROJ_NAME


def main():
    wandb.init(project=PROJ_NAME)
    print(wandb.config)
    options = ESDConfig(**wandb.config)
    train(options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps using Weights and Biases"
    )

    parser.add_argument(
        "--sweep_file", type=str, help="Path to sweep.yml file", default='scripts/sweeps.yml')

    parse_args = parser.parse_args()

    if parse_args.sweep_file is not None:
        with open(Path(parse_args.sweep_file)) as f:
            sweep_config = yaml.safe_load(f)
            print(f"Sweep config: {sweep_config}")

        sweep_id = wandb.sweep(sweep=sweep_config, project=PROJ_NAME)
        wandb.agent(sweep_id, function=main, count=10)