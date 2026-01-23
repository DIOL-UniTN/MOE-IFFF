import torch
import mlflow
import logging
import pandas as pd
from .base import BaseExp 
from utils.nn import train_epoch, eval_model


class GrowingDDT(BaseExp):
    def __init__(self):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "GrowingDDT"

    def get_config(self):
        return {
                'exp_name': self.exp_name,
                }

    def setup(self, partial_model, loader, optim, device, mconf: object, proj_name: str, 
              run_name: str | None = None):
        # MLFlow setup
        mconf.start(proj_name, run_name=None)
        # Model and optim.setup
        self.model = partial_model(
                in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                out_features=loader.out_dim
                ).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())

    def log_exp(self, metrics):
        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv')

        # Log model
        self.model.to('cpu')
        torch.save(self.model, self.out_dir/'model.pt') # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints
        mlflow.log_artifact(self.out_dir/'model.pt')
        mlflow.log_artifact(self.out_dir/'state_dict.pt')

    def run_exp(self, cfg):
        # Param. logging
        mlflow.log_params({
            'depth': self.model.depth,
            'leaf_width': self.model.leaf_width,
            'task': cfg.loader.name,
            'epochs': cfg.epochs,
            })

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(self.model, self.optim, cfg.loader.train, self.criterion, epoch, cfg.device)
            val_loss, val_acc = eval_model(self.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            self.model.to(cfg.device)

        # # Testing
        test_loss, test_acc= eval_model(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            })

        return metrics

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.loader, cfg.optim, cfg.device, cfg.mlflow, cfg.proj_name, 
                   run_name=None)
        metrics = self.run_exp(cfg)
        self.log_exp(metrics)
        self.end_run(cfg.seed)
