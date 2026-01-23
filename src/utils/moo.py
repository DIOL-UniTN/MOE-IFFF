import logging
import torch
import random
import mlflow
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.progress import ProgressBar
from pymoo.util.display.column import Column
from pymoo.core.callback import Callback

from utils.nn import train_epoch, eval_model, eval_ddt
from utils import Tree

class NASProblem(Problem):
    def __init__(self, model, optim, loader, a_sched, device, n_obj: int, 
                 n_constr: int, epochs: int, max_params: int, min_params: int, 
                 xl=None, xu=None, n_var: int = None, obj: str = "params"):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, 
                         eliminate_duplicaties=True, vtype=int, 
                         xl=xl, xu=xu,
                         )
        self.obj = obj
        self.max_params = max_params
        self.min_params = min_params
        self.epochs = epochs
        self.model = model
        self.loader = loader
        self.a_sched = a_sched
        self.optim = optim
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def _init_out(self, out, pop_size):
        out["F"] = np.empty((pop_size, self.n_obj))
        # Logging
        out["arch"] = np.zeros((pop_size, 2**(self.max_depth+1) - 1), dtype=int) - 1
        out["test"] = np.empty((pop_size))
        out["train"] = np.empty((pop_size))
        out["n_params"] = np.empty((pop_size))
        out["balance"] = np.empty((pop_size))
        out["mac"] = np.empty((pop_size))
        out["depth"] = np.empty((pop_size), dtype=int)
        out["n_leaves"] = np.empty((pop_size), dtype=int)
        out["leaf_widths"] = np.zeros((pop_size, 2**self.max_depth), dtype=int)
        out["leaf_stats"] = np.zeros((pop_size, 2**self.max_depth))
        out["leaf_macs"] = np.zeros((pop_size, 2**self.max_depth))
        return out

    def _log_out(self, out, i, arch, test_acc, train_acc, n_params, avg_macs, 
                 balance_score, depth, leaf_widths, leaf_stats, leaf_macs):
        out["arch"][i, :len(arch)] = arch
        out["test"][i] = test_acc
        out["train"][i] = train_acc
        out["n_params"][i] = n_params
        out["mac"][i] = avg_macs
        out["balance"][i] = balance_score
        out["depth"][i] = depth
        out["n_leaves"][i] = len(leaf_widths)
        out["leaf_widths"][i, :len(leaf_widths)] = leaf_widths
        out["leaf_stats"][i, :len(leaf_stats)] = leaf_stats
        out["leaf_macs"][i, :len(leaf_macs)] = leaf_macs

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        out = self._init_out(out, pop_size)

        # Create and start a process for each individual
        queues = [mp.Queue() for _ in range(pop_size)]
        processes = []
        for i in range(pop_size):
            p = mp.Process(target=self._set_and_train_individual, args=(x[i].tolist(), queues[i]))
            p.start()
            processes.append(p)
        # Wait for all processes to finish 
        for p in processes:
            p.join()
        # Collect all results
        results = [queues[i].get() for i in range(pop_size)]
        # Fill the output arrays and log results
        for i in range(pop_size):
            (val_acc, test_acc, train_acc, n_params, avg_macs, 
             balance_score, leaf_stats, arch, leaf_widths, depth, leaf_macs) = results[i]
            # To be minimized:
            out["F"][i, 0] = 1 - val_acc
            out["F"][i, 1] = n_params if self.obj == "params" else avg_macs
            # Further logging
            self._log_out(out, i, arch, test_acc, train_acc, n_params, avg_macs, 
                          balance_score, depth, leaf_widths, leaf_stats, 
                          leaf_macs)
        # Constraint
        g1 = out["F"][:, 1] > self.max_params  # max constraint
        g2 = out["F"][:, 1] < self.min_params  # min constraint
        out["G"] = np.column_stack([g1, g2]).astype(int)

    def _set_and_train_individual(self, arch: list[int], queue: mp.Queue):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        tree = self.tree(arch=arch)
        model, optim, a_sched = self._set_model(tree)
        val_acc, test_acc, train_acc, leaf_stats = self._train_individual(model, optim, 
                                                                          a_sched)
        n_params, leaf_macs = model.cal_complexity()
        avg_macs = model.get_avg_macs(leaf_stats)
        balance_score = model.get_balance_score()
        leaf_widths = model.leaf_widths
        queue.put((val_acc, test_acc, train_acc, n_params, avg_macs, 
                   balance_score, leaf_stats.tolist(), model.arch, 
                   leaf_widths, model.depth, leaf_macs))

    def _train_individual(self, model, optim, a_sched):
        for epoch in range(self.epochs):
            train_loss, train_acc = train_epoch(model, optim, self.loader.train, 
                                                self.criterion, epoch, self.device, 
                                                tqdm_disable=True)
            val_loss, val_acc = eval_model(model, self.loader.valid, self.criterion, 
                                           self.device)
            a_sched.step()
        # Testing
        test_loss, test_acc, leaf_stats = eval_ddt(model, self.loader.test, 
                                                   self.criterion, self.device)
        val_loss, val_acc = eval_model(model, self.loader.valid, self.criterion, 
                                       self.device)
        train_loss, train_acc = eval_model(model, self.loader.train, self.criterion, 
                                           self.device)
        logging.info("TEST | Acc: {:.4f}, Loss: {:.4f} | Valid | Acc: {:.4f}, Loss: {:.4f} | Train | Acc: {:.4f}, Loss: {:.4f} ".format(test_acc, test_loss, val_acc, val_loss, train_acc, train_loss))
        logging.info("TEST | leaf input distribution: {}".format(leaf_stats.tolist()))
        return val_acc, test_acc, train_acc, leaf_stats

    def _set_model(self, tree: Tree):
        # Model and optim setup
        logging.info(f"Individual with architecture: {tree.arch}")
        model = self.model(
                in_features=self.loader.in_chan*self.loader.in_size[0]*self.loader.in_size[1], 
                out_features=self.loader.out_dim, tree=tree,
                ).to(self.device)
        logging.info(f"Individual with fixed architecture: {model.arch} and leaf widths: {model.leaf_widths}")
        optim = self.optim(model.parameters())
        a_sched = self.a_sched(model.alpha)
        return model, optim, a_sched

class NASDDTProblem(NASProblem):
    def __init__(self, model, optim, loader, a_sched, device, n_obj: int, 
                 n_constr: int, epochs: int, max_params: int, min_params: int, 
                 tree: Tree, n_var: int = None, obj: str = "params"):
        t = tree()
        xl=np.array([t.get_node_id("router")] + 
                     [t.get_node_id("none")]*(t.max_nodes-1))
        xu=np.array([t.get_node_id("router")] +  #TODO: Check -> Not sure if necessary yet
                     [t.max_width]*(t.max_nodes-1))
        n_var = t.max_nodes 
        super().__init__(model, optim, loader, a_sched, device, n_obj, 
                         n_constr, epochs, max_params, min_params, 
                         xl=xl, xu=xu, n_var=n_var, obj=obj)
        self.tree = tree
        self.max_depth = t.max_depth
        self.max_nodes = t.max_nodes
        self.max_width = t.max_width
        self.max_params = max_params
        self.min_params = min_params

class NASDDTProblemFixLeaf(NASProblem):
    def __init__(self, model, optim, loader, a_sched, device, n_obj: int, 
                 n_constr: int, epochs: int, max_params: int, min_params: int, 
                 tree: Tree, obj: str = "params"):
         t = tree()
         xl=np.array([t.get_node_id("router")] + 
                     [t.get_node_id("none")]*(t.max_nodes-1) + [1]), 
         xu=np.array([t.get_node_id("router")] +  #TODO: Check -> Not sure if necessary yet
                     [t.get_node_id("leaf")]*(t.max_nodes-1) + [t.max_width])
         super().__init__(model, optim, loader, a_sched, device, n_obj, 
                          n_constr, epochs, max_params, min_params, 
                          tree, xl=xl, xu=xu, n_var=t.max_nodes+1, 
                          obj=obj)

    def _set_and_train_individual(self, arch: list[int], queue: mp.Queue):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        leaf_width = arch[-1]
        arch = arch[:-2]
        tree = self.tree(arch=arch)
        tree() # Fix
        tree.arch = [n if n <= 0 else leaf_width for n in tree.arch] # Fix leaves

        model, optim, a_sched = self._set_model(tree)
        val_acc, test_acc, train_acc, leaf_stats = self._train_individual(model, optim, 
                                                                          a_sched)
        n_params, leaf_macs = model.cal_complexity()
        avg_macs = model.get_avg_macs(leaf_stats)
        balance_score = model.get_balance_score()
        leaf_widths = model.leaf_widths
        queue.put((val_acc, test_acc, train_acc, n_params, avg_macs, 
                   balance_score, leaf_stats.tolist(), model.arch, 
                   leaf_widths, model.depth, leaf_macs))

class DDTRandomImbalancedSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        archs = [problem.tree()._gen_random_arch() for _ in range(n_samples)]
        return np.array(archs)

class DDTRandomBalancedSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        archs = [problem.tree()._gen_random_balanced_arch() for _ in range(n_samples)]
        return np.array(archs)

class DDTRandomBalancedSamplingFixLeaf(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        archs = [problem.tree()._gen_random_balanced_arch() for _ in range(n_samples)]
        archs = np.array(archs)
        leaf_widths = archs[archs > 0][:archs.shape[0]]
        leaf_widths = leaf_widths.reshape(-1, 1)
        archs[archs > 0] = 1
        archs = np.hstack((archs, leaf_widths))
        return archs

class DDTMetricsLogger(Callback):
    def notify(self, algorithm):
        pop = algorithm.pop
        # Logging Objectives
        mlflow.log_param(f"val_accs_{algorithm.n_gen-1}", str((1 - pop.get("F")[:, 0]).tolist()).replace(", ", ","))
        # Further logging
        # Further logging
        archs = pop.get("arch")
        max_archs = (archs != -2).sum(axis=1).max()
        archs = archs[:, :max_archs]
        mlflow.log_param(f"archs_{algorithm.n_gen-1}", str(archs.astype(int).tolist()).replace(", ", ","))
        mlflow.log_param(f"tests_{algorithm.n_gen-1}", str(pop.get("test").tolist()).replace(", ", ","))
        mlflow.log_param(f"trains_{algorithm.n_gen-1}", str(pop.get("train").tolist()).replace(", ", ","))
        mlflow.log_param(f"n_params_{algorithm.n_gen-1}", str(pop.get("n_params").tolist()).replace(", ", ","))
        mlflow.log_param(f"macs_{algorithm.n_gen-1}", str(pop.get("mac").tolist()).replace(", ", ","))
        mlflow.log_param(f"balances_{algorithm.n_gen-1}", str(pop.get("balance").tolist()).replace(", ", ","))
        mlflow.log_param(f"depths_{algorithm.n_gen-1}", str(pop.get("depth").tolist()).replace(", ", ","))
        mlflow.log_param(f"n_leaves_{algorithm.n_gen-1}", str(pop.get("n_leaves").tolist()).replace(", ", ","))

        leaf_widths = pop.get("leaf_widths")
        leaf_macs = pop.get("leaf_macs")
        leaf_stats = pop.get("leaf_stats")
        max_leaves = (leaf_widths != 0).sum(axis=1).max()
        leaf_widths = leaf_widths[:, :max_leaves]
        leaf_stats = leaf_stats[:, :max_leaves]
        leaf_macs = leaf_macs[:, :max_leaves]
        leaf_stats = leaf_stats[:, :max_leaves]
        mlflow.log_param(f"leaf_widths_{algorithm.n_gen-1}", str(leaf_widths.tolist()).replace(", ", ","))
        mlflow.log_param(f"leaf_stats_{algorithm.n_gen-1}", str(leaf_stats.tolist()).replace(", ", ","))
        mlflow.log_param(f"leaf_macs_{algorithm.n_gen-1}", str(leaf_macs.tolist()).replace(", ", ","))

class DDTOutput(MultiObjectiveOutput):
    def __init__(self):
        super().__init__()
        self.val_acc = Column("val_acc", width=8)
        self.n_params = Column("params", width=8)
        self.test_acc = Column("test_acc", width=8)
        self.macs = Column("macs", width=8)
        self.depth = Column("depth", width=8)
        self.balance = Column("balance", width=8)
        self.columns += [self.val_acc, self.n_params, self.test_acc, 
                         self.macs, self.depth, self.balance]

    def update(self, algorithm):
        super().update(algorithm)
        self.val_acc.set(np.mean(1-algorithm.pop.get("F")[:,0]))
        self.n_params.set(algorithm.pop.get("n_params").mean())
        self.macs.set(algorithm.pop.get("mac").mean())
        self.test_acc.set(algorithm.pop.get("test").mean())
        self.depth.set(algorithm.pop.get("depth").mean())
        self.balance.set(algorithm.pop.get("balance").mean())
        
    def text(self):
        regex = " | ".join(["{}"] * len(self.columns))
        return regex.format(*[col.text() for col in self.columns])

class DDTDisplay(Callback):
    def __init__(self, output=DDTOutput(), progress=True, verbose=True):
        super().__init__()
        self.output = output
        self.verbose = verbose
        self.progress = ProgressBar() if progress else None
        formatter = logging.Formatter("%(message)s")

    def update(self, algorithm, **kwargs):
        output, progress = self.output, self.progress

        if self.verbose and output:
            text = ""
            header = not output.is_initialized
            output(algorithm)

            if header:
                logging.info(output.header(border=False))
            text += output.text()
            logging.info(text)

        if progress:
            perc = algorithm.termination.perc
            progress.set(perc)

    def finalize(self):

        if self.progress:
            self.progress.close()
