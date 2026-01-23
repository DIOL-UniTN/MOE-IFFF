import logging
from .base import BaseExp 

class MyExp(BaseExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "MyExp"

    def get_config(self):
        return {
                'exp_name': self.exp_name,
                }

    def setup(self, mconf: object, proj_name: str, 
              run_name: str | None = None):
        # MLFlow setup
        mconf.start(proj_name, run_name=None)

    def run_exp(self, cfg):
        pass

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.mlflow, cfg.proj_name, run_name=None)
        self.run_exp(cfg)
        self.end_run(cfg.seed)
