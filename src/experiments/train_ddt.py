import torch
import mlflow
import logging
import pandas as pd
from typing import Optional
from .base import BaseExp 
from utils.nn import train_epoch, eval_model, eval_ddt

class TrainDDT(BaseExp):
    def __init__(self, arch: Optional[int]):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "TrainDDT"
        self.arch = str(arch)

    def get_config(self):
        exp_config = {'exp_name': self.exp_name,
                      'arch': self.arch,
                      }
        return (exp_config | self.model.get_config() | self.loader.get_config() | 
                self.mlflow.get_config() | self.a_sched.get_config())

    def setup(self, partial_model, loader, optim, a_sched: object, device, 
              mconf: object, proj_name: str, run_name: Optional[str] = None):
        # MLFlow setup
        mconf.start(proj_name, run_name=None)
        self.mlflow = mconf

        # Model and optim.setup
        self.model = partial_model(
                in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                out_features=loader.out_dim, arch=self.arch
                ).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.loader = loader
        self.a_sched = a_sched(self.model.alpha)

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

    def run_exp(self, epochs, device):
        # Param. logging
        mlflow.log_params({
            'epochs': epochs,
            })

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, device)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            self.model.to(device)
            self.a_sched.step()

        # Testing
        test_loss, test_acc, leaf_stats = eval_ddt(self.model, self.loader.test, 
                                                     self.criterion, device)
        train_eval_loss, train_eval_acc, train_leaf_stats = eval_ddt(self.model, self.loader.train, 
                                                     self.criterion, device) 
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        logging.info("TEST | leaf input distribution: {}".format(leaf_stats.tolist()))
        logging.info("TRAIN | eval_acc: {:.4f}, eval_loss: {:.4f}, ".format(train_eval_acc, train_eval_loss))
        logging.info("TRAIN | leaf input distribution: {}".format(train_leaf_stats.tolist()))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_eval_loss': train_eval_loss,
            'train_eval_acc': train_eval_acc,
            'avg_macs': self.model.get_avg_macs(leaf_stats),
            'macs': self.model.get_avg_macs(leaf_stats),
            })

        return metrics

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.loader, cfg.optim, cfg.a_sched, cfg.device, cfg.mlflow, 
                   cfg.proj_name, run_name=None)
        self.start_run(cfg.seed)
        metrics = self.run_exp(cfg.epochs, cfg.device)
        self.log_exp(metrics)
        self.end_run()
