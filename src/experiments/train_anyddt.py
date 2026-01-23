import torch
import mlflow
import logging
import pandas as pd
from typing import Optional
import torch.multiprocessing as mp
from .base import BaseExp 
from utils.nn import train_epoch, eval_model, eval_ddt

class TrainAnyDDT(BaseExp):
    def __init__(self, tree: list[int], n_runs: int = 1):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "TrainAnyDDT"
        self.tree = tree
        self.n_runs = n_runs

    def get_config(self):
        exp_config = {'exp_name': self.exp_name,
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
                out_features=loader.out_dim, tree=self.tree).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.loader = loader
        self.a_sched = a_sched(self.model.alpha)
        self.device = device

        self.p_model = partial_model
        self.p_optim = optim
        self.p_a_sched = a_sched

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

    def run_exp(self, epochs):
        if self.n_runs == 1:
            self.run_exp_single(epochs)
        else:
            self.run_exp_parallel(epochs)

    def train_model(self, epochs, queue: mp.Queue):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = self.p_model(
                in_features=self.loader.in_chan*self.loader.in_size[0]*self.loader.in_size[1], 
                out_features=self.loader.out_dim, tree=self.tree
                ).to(self.device)
        optim = self.p_optim(model.parameters())
        a_sched = self.p_a_sched(model.alpha)
        # Training
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, optim, self.loader.train, 
                                                self.criterion, epoch, self.device, 
                                                tqdm_disable=True)
            a_sched.step()

        # Testing
        val_loss, val_acc = eval_model(model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
        test_loss, test_acc, leaf_stats = eval_ddt(model, self.loader.test, 
                                                     self.criterion, self.device)
        train_eval_loss, train_eval_acc, train_leaf_stats = eval_ddt(model, self.loader.train, 
                                                     self.criterion, self.device) 
        avg_macs = model.get_avg_macs(leaf_stats)
        queue.put((test_acc, test_loss, leaf_stats.tolist(), val_acc, val_loss, train_acc, 
                  train_loss, train_eval_acc, train_eval_loss, avg_macs))

    def run_exp_single(self, epochs):
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
            train_loss, train_acc = train_epoch(self.model, self.optim, self.loader.train, self.criterion, epoch, self.device)
            val_loss, val_acc = eval_model(self.model, self.loader.valid, self.criterion, self.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            self.model.to(self.device)
            self.a_sched.step()

        # Testing
        test_loss, test_acc, leaf_stats = eval_ddt(self.model, self.loader.test, 
                                                     self.criterion, self.device)
        train_eval_loss, train_eval_acc, train_leaf_stats = eval_ddt(self.model, self.loader.train, 
                                                     self.criterion, self.device) 
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        logging.info("TEST | leaf input distribution: {}".format(leaf_stats.tolist()))
        logging.info("TRAIN | eval_acc: {:.4f}, eval_loss: {:.4f}, ".format(train_eval_acc, train_eval_loss))
        logging.info("TRAIN | leaf input distribution: {}".format(train_leaf_stats.tolist()))

        mlflow.log_param(f"leaf_stats", str(leaf_stats))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_eval_loss': train_eval_loss,
            'train_eval_acc': train_eval_acc,
            'avg_macs': self.model.get_avg_macs(leaf_stats),
            'macs': self.model.get_avg_macs(leaf_stats),
            })

        return metrics

    def run_exp_parallel(self, epochs):
        queues = [mp.Queue() for _ in range(self.n_runs)]
        processes = []
        for i in range(self.n_runs):
            p = mp.Process(target=self.train_model, args=(epochs, queues[i]))
            p.start()
            processes.append(p)
        # Wait for all processes to finish 
        for p in processes:
            p.join()
        # Collect all results
        results = [q.get() for q in queues]

        test_accs, test_losses = [], []
        train_accs, train_losses = [], []
        train_eval_accs, train_eval_losses = [], []
        val_accs, val_losses = [], []
        all_leaf_stats = []
        for i, res in enumerate(results):
            (test_acc, test_loss, leaf_stats, val_acc, val_loss, train_acc, 
             train_loss, train_eval_acc, train_eval_loss, avg_macs) = res
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            train_eval_accs.append(train_eval_acc)
            train_eval_losses.append(train_eval_loss)
            all_leaf_stats.append(leaf_stats)
        logging.info("TEST | acc: {}".format(test_accs, test_losses))
        logging.info("VAL | acc: {}".format(val_accs, val_losses))
        logging.info("TRAIN | acc: {}".format(train_accs, train_losses))
        logging.info("TRAIN | eval_acc: {}".format(train_eval_accs, train_eval_losses))
        logging.info("TEST | leaf input distribution: {}".format(all_leaf_stats))

        # Param. logging
        metrics = {
            'val_mean': torch.tensor(val_accs).mean().item(),
            'val_std': torch.tensor(val_accs).std().item(),
            'test_mean': torch.tensor(test_accs).mean().item(),
            'test_std': torch.tensor(test_accs).std().item(),
            'test_losses': test_losses,
            'test_accs': test_accs,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'train_eval_losses': train_eval_losses,
            'train_eval_accs': train_eval_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'avg_macs': avg_macs,
            'macs': avg_macs,
            'epochs': epochs,
            'leaf_stats': str(leaf_stats),
            }
        mlflow.log_params(metrics)
        return metrics

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}, #Runs: {self.n_runs}")
        self.setup(cfg.model, cfg.loader, cfg.optim, cfg.a_sched, cfg.device, cfg.mlflow, 
                   cfg.proj_name, run_name=None)
        self.start_run(cfg.seed)
        metrics = self.run_exp(cfg.epochs)
        self.log_exp(metrics)
        self.end_run()
