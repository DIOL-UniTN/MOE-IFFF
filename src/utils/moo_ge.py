import logging
import torch
import random
import mlflow
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from utils import Tree
from utils.grammar import DDTGrammar
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import Sampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.progress import ProgressBar
from pymoo.util.display.column import Column
from pymoo.core.callback import Callback

from utils.nn import train_epoch, eval_model, eval_ddt

class NASGEProblem(Problem):
    def __init__(self, model, optim, loader, a_sched, device, n_obj: int, 
                 n_constr: int, epochs: int, max_params: int, min_params: int,
                 grammar: DDTGrammar, obj: str = "params"):
        super().__init__(n_var=grammar.len_gene, n_obj=n_obj, n_constr=n_constr, 
                         eliminate_duplicaties=True, vtype=int, 
                         xl=0, xu=grammar.high,
                         )
        self.obj = obj
        self.grammar = grammar
        self.max_depth = grammar.max_depth
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
        self._init_out(out, pop_size)

        # Create and start a process for each individual
        genotypes = [self.grammar.gene_to_tree(x[i].tolist()) for i in range(pop_size)]
        queues = [mp.Queue() for _ in range(pop_size)]
        processes = []

        for i in range(pop_size):
            arch = self.grammar.gene_to_tree(x[i].tolist())
            p = mp.Process(target=self._set_and_train_individual, args=(arch, queues[i]))
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
        g2 = out["F"][:, 1] < self.min_params  # max constraint
        out["G"] = np.column_stack([g1, g2]).astype(int)

    def _set_and_train_individual(self, arch: list[int], queue: mp.Queue):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        tree = Tree(arch=arch)
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

class NASDDTGEProblem(NASGEProblem):
    def __init__(self, model, optim, loader, a_sched, device, n_obj: int, 
                 n_constr: int, epochs: int, max_params: int, min_params: int,
                 grammar: DDTGrammar, obj: str = "params"):
        super().__init__(model, optim, loader, a_sched, device, n_obj, 
                         n_constr, epochs, max_params, min_params,
                         grammar, obj=obj)

class DDTGERandomImbalancedSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        gens = []
        high = problem.grammar.high
        for i in range(n_samples):
            gen = [random.randint(0, high) for _ in range(problem.n_var)]
            gens.append(gen)
        return np.array(gens)

class DDTGERandomBalancedSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        gens = []
        max_depth = problem.max_depth
        high = problem.grammar.high
        n_var = problem.n_var
        for i in range(n_samples):
            depth = random.randint(1, max_depth)
            n_routers, n_leaves = 2**(depth-1) - 1, 2**depth # Since 1 root already there "-2" 
            gen = [3 * random.randint(0, high//3) for _ in range(n_routers)] # routers
            gen += [3 * random.randint(0, high//3-1)+2 for _ in range(n_leaves)] # routers
            gen += [random.randint(0, high) for _ in range(n_routers+n_leaves, n_var)] # remainig (doesnt matter, just random)
            gens.append(gen)
        return np.array(gens)
