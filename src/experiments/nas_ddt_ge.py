import shutil
import numpy as np
import pickle
import mlflow
import logging
from functools import partial
from pathlib import Path
from .base import BaseExp 

from utils.mlflow import MLFlow

from utils.grammar import DDTGrammar
from utils.moo import DDTDisplay, DDTMetricsLogger
from utils.moo_ge import NASDDTGEProblem, DDTGERandomImbalancedSampling, DDTGERandomBalancedSampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo import operators
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.repair.rounding import RoundingRepair

MUTATIONS = {
        "pm": partial(operators.mutation.pm.PM, eta=3.0, vtype=float, repair=RoundingRepair()),
        "bflip": partial(BitflipMutation, vtype=float, repair=RoundingRepair()),
        }
CROSSOVERS = {
        "sbx": partial(operators.crossover.sbx.SBX, eta=3.0, vtype=float, repair=RoundingRepair()),
        "p": partial(PointCrossover, vtype=float, repair=RoundingRepair()),
        }
SAMPLINGS = {
        "imbalanced": DDTGERandomImbalancedSampling,
        "balanced": DDTGERandomBalancedSampling,
        }
class NASDDTGE(BaseExp):
    def __init__(self, n_gens: int, pop_size: int, max_params: int, min_params: int,
                 crossover, mutation, grammar: DDTGrammar, fix_leaf: bool, sampling: str, 
                 obj: str):
        super().__init__()  # Initialize BaseExp
        self.obj = obj
        self.fix_leaf = fix_leaf
        self.sampling = sampling
        self.grammar = grammar
        self.exp_name = "NASDDTGE"
        self.max_params = max_params
        self.min_params = min_params
        self.n_gens, self.pop_size = n_gens, pop_size
        if crossover.type == "p":
            crossover = CROSSOVERS[crossover.type](prob=crossover.p, 
                                                   n_points=crossover.n_points)
        else:
            crossover = CROSSOVERS[crossover.type](prob=crossover.p)
        self.algorithm = NSGA2(pop_size=self.pop_size, 
                               sampling=SAMPLINGS[sampling](),
                               crossover=crossover,
                               mutation=MUTATIONS[mutation.type](prob=mutation.p),
                               eliminate_duplicates=True,
                               )

    def get_config(self):
        exp_config = {'exp_name': self.exp_name,
                      'n_gens': self.n_gens,
                      'obj': self.obj,
                      'sampling': self.sampling,
                      'fix_leaf': self.fix_leaf,
                      'n_obj': self.n_obj,
                      'n_var': self.n_var,
                      'max_params': self.max_params,
                      'min_params': self.min_params,
                      'pop_size': self.pop_size,
                      'seed': self.seed,
                      }
        return (exp_config | self.mlflow.get_config() | self.loader.get_config() | self.grammar.get_config())

    def setup(self, partial_model, optim, loader, a_sched, epochs: int, device, 
              mconf: MLFlow, proj_name: str, seed: int):
        # MLFlow setup
        mconf.start(proj_name, run_name=None)
        self.mlflow = mconf

        self.seed = seed
        self.loader = loader
        self.n_obj = 2 # acc | params
        self.n_constr = 2 # max n. params
        self.n_var = self.grammar.len_gene

        # Problem setup
        self.callback = DDTMetricsLogger()
        self.problem = NASDDTGEProblem(partial_model, optim, loader, a_sched, device, 
                                       self.n_obj, n_constr=2, epochs=epochs, 
                                       max_params=self.max_params, min_params=self.min_params, 
                                       grammar=self.grammar, obj=self.obj)

    def log_exp(self, res):
        # Save results
        attrs = list(res.history[0].__dict__.keys())
        for hist in res.history:
            for attr in attrs:
                if attr not in ['pop', 'off', 'opt']:
                    delattr(hist, attr) # We don't want to save everything. Too big and requires the code! (in GBs)
            setattr(hist, "task", self.loader.name) # We can add other attributes we will need
        with open(self.out_dir/"history.dump", "wb") as f:
            pickle.dump(res.history, f)
        mlflow.log_artifact(str(self.out_dir/"history.dump"))
        # Log final Pareto front
        mlflow.log_param(f"final_val_accs", str((1 - res.F[:, 0]).tolist()).replace(", ", ","))
        mlflow.log_param(f"final_n_params", str((res.F[:, 1]).astype(int).tolist()).replace(", ", ","))
        mlflow.log_param(f"final_archs", str(res.opt.get("arch").tolist()).replace(", ", ","))
        mlflow.log_param(f"final_macs", str(res.opt.get("mac").tolist()).replace(", ", ","))
        # Scatter plot
        plot = Scatter(title=f"TASK: {self.loader.name}, MAX PARAMS: {self.max_params}, GENE BITS: {self.grammar.gene_bits}, GENES LEN: {self.grammar.len_gene}, MAX WIDTH: {self.grammar.max_leaf_width}, MAX DEPTH: {self.grammar.max_depth}, POP SIZE: {self.pop_size}, N GENS: {self.n_gens}, N OBJ: {self.n_obj}", 
                       labels=["Number of Parameters", "Val. Accuracy"])
        # All individuals
        F_hist = np.vstack([r.pop.get("F") for r in res.history])
        F_hist[:, 0] = 1 - F_hist[:, 0]  # Convert error to acc
        plot.add(F_hist[:, [1, 0]], facecolor="none", edgecolor="blue")
        # Pareto front
        pareto = np.stack([1 - res.F[:, 0], res.F[:, 1]], axis=1)
        sorted_pareto_indices = np.argsort(pareto[:, 0])  # Sort by first column
        pareto = pareto[sorted_pareto_indices]
        plot.add(pareto[:, [1, 0]], plot_type="line", marker="x", color="red", alpha=0.7)
        # Save
        plot.save(self.out_dir/"scatter.png")
        mlflow.log_artifact(str(self.out_dir/"scatter.png"))
        # Move mlflow logged files to ./data folder
        run = mlflow.active_run()
        assert run is not None, "Expected 'run' to be non-None, but got None"
        assert run.info.artifact_uri is not None, "Expected 'artifact_uri' to be non-None, but got None"

        mlflow.log_param("mlflow_run_id", run.info.run_id)

        tracking_uri = Path(mlflow.get_tracking_uri().replace("file:", ""))
        artifact_uri = Path(run.info.artifact_uri.replace("file:", ""))
        if not (tracking_uri.parent/f"artifacts/{self.mlflow.exp}").is_dir():
            (tracking_uri.parent/f"artifacts/{self.mlflow.exp}").mkdir(parents=True, 
                                                                       exist_ok=True)
        (artifact_uri/f"scatter.png").replace(tracking_uri.parent/
                                              f"artifacts/{self.mlflow.exp}"/
                                              f"scatter_{run.info.run_id}.png")
        (artifact_uri/f"history.dump").replace(tracking_uri.parent/
                                               f"artifacts/{self.mlflow.exp}"/
                                               f"history_{run.info.run_id}.dump")
        log_files = list(self.out_dir.glob("*.log"))
        for i, log_file in enumerate(log_files):
            shutil.copy2(log_file, tracking_uri.parent/f"artifacts/{self.mlflow.exp}"/
                         f"{run.info.run_id}_{i}.log")

    def run_exp(self):
        res = minimize(self.problem, self.algorithm, seed=self.seed, 
                       termination=('n_gen', self.n_gens), 
                       save_history=True,
                       display=DDTDisplay(progress=False), callback=self.callback)
        return res

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.optim, cfg.loader, cfg.a_sched, cfg.epochs, cfg.device, 
                   cfg.mlflow, cfg.proj_name, cfg.seed)
        self.start_run(cfg.seed)
        res = self.run_exp()
        self.log_exp(res)
        self.end_run()
