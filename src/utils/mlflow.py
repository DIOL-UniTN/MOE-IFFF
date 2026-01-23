import os
import logging
from pathlib import Path
import mlflow
import dagshub
from typing import Optional

class MLFlow:
    def __init__(self, username: str, token:str, exp: str, 
                 server: str, env: object):
        self.token = token
        self.exp = exp
        self.server = server
        self.username = username
        self.env = env

    def start(self, proj_name:str, 
              run_name: Optional[str] = None):
        logging.info(
            f"MLFlow server is starting on {self.server.upper()} "
            f"with experiment {self.exp}..."
        )
        # Set mlflow env. variables
        for (var, val) in zip(self.env.vars, self.env.vals):
            os.environ[var] = str(val)
        if self.server == "local":
            self.start_local(proj_name, run_name)
        else:
            self.start_dagshub(proj_name, run_name)

    def start_local(self, proj_name:str, run_name: Optional[str] = None):
        mlflow.set_experiment(self.exp)
        mlflow.end_run()
        mlflow.set_experiment(self.exp)
        mlflow.start_run(run_name=run_name)
        run = mlflow.active_run()
        logging.info(f"MLFlow run ID: {run.info.run_id}, status: {run.info.status}")

    def start_dagshub(self, proj_name:str, run_name: Optional[str] = None):
        dagshub.init(proj_name, self.username)
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = self.token
        mlflow.set_experiment(self.exp)
        mlflow.start_run(run_name=run_name)
        run = mlflow.active_run()
        logging.info(f"MLFlow run ID: {run.info.run_id}, status: {run.info.status}")

    def get_config(self):
        return {
                "mlflow_exp": self.exp,
                "mlflow_server": self.server,
                }
