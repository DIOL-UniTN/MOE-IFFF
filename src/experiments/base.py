import hydra
import mlflow
import logging
from pathlib import Path
from abc import ABC, abstractmethod

class BaseExp(ABC):
    def __init__(self):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    def start_run(self, seed: int):
        mlflow.log_param('seed', seed)
        mlflow.log_param("out_dir", self.out_dir)
        mlflow.log_params(self.get_config())
        mlflow.log_artifact(self.overrides_config)

    def end_run(self):
        mlflow.log_param('success', True)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        run = mlflow.active_run()
        mlflow.end_run()
        finished_run = mlflow.get_run(run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, "
                     f"status: {finished_run.info.status}")

    def end_failed_run(self, error, seed: int):
        logging.info("Failed!")
        logging.exception(error)
        mlflow.log_param('success', False)

        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        run = mlflow.active_run()
        mlflow.end_run()
        finished_run = mlflow.get_run(run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, "
                     f"status: {finished_run.info.status}")

    @abstractmethod
    def run_exp(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def main(self, cfg):
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failed_run(e, cfg.seed)
