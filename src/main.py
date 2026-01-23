import os
import random
import hydra
import mlflow
import torch
import logging
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
from hydra.experimental.callback import Callback

class MetricsCallback(Callback):
    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        exp_name = config["mlflow"]["exp"]
        exp = mlflow.get_experiment_by_name(exp_name)
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        df.columns = [col.replace("metrics.", "").replace("params.", "") for col in df.columns]

        logging.info(f"Jobs ended, exporting as CSV...")
        csv_dir = Path("./data/csv_files/")
        if not csv_dir.is_dir():
            csv_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv((csv_dir/exp_name).with_suffix(".csv"))
        
@dataclass
class Main:
    proj_name: str
    run_name: str
    seed: int
    debug: bool
    exp: object
    a_sched: object
    mlflow: object
    model: torch.nn.Module
    optim: torch.optim.Adam
    loader: object
    device: torch.device
    epochs: int

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    # Init RNGs
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cfg = instantiate(cfg)

    # Run experiment
    if cfg.debug > 0:
        os.environ["HYDRA_FULL_ERROR"] = "1" # In debug mode
        cfg.exp.run(cfg)
    else:
        cfg.exp.main(cfg)

if __name__ == "__main__":
    main()
