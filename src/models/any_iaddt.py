import torch 
import math
from torch import nn
from typing import Optional, Union
from utils.tree import Tree

from utils.awareness import Sign

class AnyIADDT(nn.Module):
    def __init__(self, in_features: int, out_features: int, tree: Tree):
        super().__init__()
        self.name = "AnyIADDT"
        self.in_features = in_features
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.sign = Sign.apply
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False) # Awareness parameter

        # Main configurations
        self.arch = tree()
        self.arch_depth = tree.gen_tree_depths()
        leaf_ids, leaf_depths = tree.get_leaf_ids()
        self.leaf_ids = nn.Parameter(torch.tensor(leaf_ids, dtype=torch.long).view(-1, 1), requires_grad=False)
        self.leaf_depths = nn.Parameter(torch.tensor(leaf_depths, dtype=torch.long), requires_grad=False)
        self.leaf_widths = tree.get_leaf_widths()
        self.leaf_widths_tensor = nn.Parameter(torch.tensor(self.leaf_widths, dtype=torch.float), requires_grad=False)
        self.max_width = max(self.leaf_widths)

        # Unbalanced (real) tree configurations
        self.n_routers = tree.router_count(self.arch)
        self.n_leaves = tree.leaf_count(self.arch)

        # Balanced tree configurations
        self.depth = len(self.arch_depth) - 1
        self.n_leavesb = 2 ** self.depth
        self.n_routersb = 2 ** self.depth - 1

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.router_weights = nn.Parameter(torch.empty((self.n_routersb, in_features)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.router_biases = nn.Parameter(torch.empty((self.n_routersb, 1)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.in_features)
        self.w1s = nn.Parameter(torch.empty((self.n_leavesb, in_features, self.max_width)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leavesb, self.max_width)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leavesb, self.max_width, out_features)).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leavesb, out_features)).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def mask_leaves(self):
        for i, l in enumerate(self.leaf_ids):
            w = self.leaf_widths[i]
            self.w1s.data[l, :, w:] = 0
            self.b1s.data[l, w:] = 0
            self.w2s.data[l, w:, ] = 0

    def forward(self, x: torch.Tensor, leaf_ids: bool = False):
        x = x.view(len(x), -1)
        if self.training:
            return  self.training_forward(x)
        else:
            return self.eval_forward(x, leaf_ids)

    def training_forward(self, x: torch.Tensor):
        batch_size, device = x.shape[0], x.device

        prob_mixture = torch.ones((batch_size, self.n_leavesb), device=device)
        dec_mixture = torch.ones((batch_size, self.n_leavesb), device=device)
        # Router nodes
        for d in range(self.depth):
            platform, next_platform = 2 ** d - 1, 2 ** (d+1) - 1

            n_routersb = 2 ** d
            w = self.router_weights[platform:next_platform]   
            b = self.router_biases[platform:next_platform]

            router_logits = torch.matmul(x, w.transpose(0, 1)) + b.transpose(0, 1)

            # Probs
            prob = (torch.sigmoid(router_logits).unsqueeze(-1))
            probs = torch.cat((1 - prob, prob), dim=-1)

            prob_mixture_modifier = probs.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)
            prob_mixture = prob_mixture.view(-1, 2 * n_routersb, self.n_leavesb // (2 * n_routersb))
            prob_mixture = prob_mixture * prob_mixture_modifier 
            prob_mixture = prob_mixture.flatten(start_dim=1, end_dim=2)

            # Decs
            dec = (self.sign(router_logits).unsqueeze(-1))
            decs = torch.cat((1 - dec, dec), dim=-1)

            dec_mixture_modifier = decs.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)
            dec_mixture = dec_mixture.view(-1, 2 * n_routersb, self.n_leavesb // (2 * n_routersb))
            dec_mixture = dec_mixture * dec_mixture_modifier 
            dec_mixture = dec_mixture.flatten(start_dim=1, end_dim=2)

        # Leave nodes
        self.mask_leaves() # Mask neurons 
        outputs = torch.empty((batch_size, self.n_leaves, self.out_features), 
                                 device=device)
        for i in range(self.n_leaves):
            l, d = self.leaf_ids[i], self.leaf_depths[i]
            logits = torch.matmul(x, self.w1s[l].transpose(0, 1).flatten(1, 2))
            logits = logits.view(batch_size, 1, self.max_width)
            logits += self.b1s[l].view(1, self.max_width)
            leaf_activations = self.activation(logits)
            outputs[:, i] = (torch.matmul(
                leaf_activations,
                self.w2s[l]
            ) + self.b2s[l]).squeeze()

            n_l = 2 ** (self.depth - d) # Number of leaves sharing the same parent
            leaf_prob_mixture = prob_mixture[:, l:(l+n_l)].sum(dim=1, keepdim=True)
            leaf_dec_mixture = dec_mixture[:, l:(l+n_l)].sum(dim=1, keepdim=True)
            outputs[:, i] *= (leaf_prob_mixture * self.alpha 
                              + leaf_dec_mixture * (1-self.alpha))

        outputs = outputs.sum(dim=1)
        return outputs

    def eval_forward(self, x: torch.Tensor, leaf_ids: bool = False) -> torch.Tensor:
        batch_size, device = x.shape[0], x.device

        # Router nodes
        routers = torch.zeros((batch_size,), dtype=torch.long, device=device)
        for d in range(self.depth):
            w = self.router_weights.index_select(dim=0, index=routers)
            b = self.router_biases.index_select(dim=0, index=routers)
            logits = torch.bmm(x.unsqueeze(1), w.unsqueeze(-1)).squeeze(-1) + b
            choices = (logits.squeeze(-1) >= 0)

            platform, next_platform = 2 ** d - 1, 2 ** (d+1) - 1
            routers = ((routers - platform) * 2  + choices + next_platform)

        # Leaf nodes
        self.mask_leaves() # Mask neurons 
        leaves = routers - next_platform              # (batch_size,)
        outputs = torch.empty((batch_size, self.out_features), device=device)
        for i in range(self.n_leaves):
            l, d = self.leaf_ids[i], self.leaf_depths[i]
            n_l = 2 ** (self.depth - d) # Number of leaves sharing the same parent
            leaf_indices, = torch.where((leaves >= l) & (leaves < (l+n_l)))

            logits = torch.matmul(x[leaf_indices], self.w1s[l]) + self.b1s[l].unsqueeze(0)                                               # (1, self.max_width)
            activations = self.activation(logits)           # (1, self.max_width)
            logits = torch.matmul(activations, self.w2s[l]) + self.b2s[l].unsqueeze(0)                                               # (1, self.max_width)
            outputs[leaf_indices] = logits

        if leaf_ids:
            return outputs, leaves
        return outputs

    def cal_complexity(self) -> tuple[float, float, float]: # Return #params, min MAC, max MAC
        # Params
        leaf_params = ((self.in_features + self.out_features) * self.leaf_widths_tensor).sum().item()
        n_params = self.n_routers * self.in_features + leaf_params
        # MACs
        leaf_macs = (self.leaf_depths + self.leaf_widths_tensor) * self.in_features
        return n_params, leaf_macs.tolist()

    def get_config(self):
        n_params, leaf_macs = self.cal_complexity()
        return {
                'model_name': self.name,
                'depth': self.depth,
                'max_width': self.max_width,
                'n_routers': self.n_routers,
                'n_leaves': self.n_leaves,
                'n_routers': self.n_routers,
                'arch': self.arch,
                'arch depth': self.arch_depth,
                'in_features': self.in_features,
                'out_features': self.out_features,
                'n_params': n_params,
                'min_macs': min(leaf_macs),
                'max_macs': max(leaf_macs),
                'leaf_ids': self.leaf_ids.tolist(),
                }

    def get_leaf_stats(self, leaf_ids):
        # Returns how many inputs each leaf received in the current batch
        return torch.tensor([(leaf_ids == i).sum() for i in self.leaf_ids])
    
    def get_avg_macs(self, leaf_stats) -> float:
        leaf_macs = (self.leaf_depths + self.leaf_widths_tensor) * self.in_features
        avg_macs = (leaf_macs * leaf_stats).sum().item()
        return avg_macs

    def get_balance_score(self) -> float:
        # std. of depths
        leaf_depths = torch.log2(self.leaf_ids+1).floor()
        return leaf_depths.std().item()

