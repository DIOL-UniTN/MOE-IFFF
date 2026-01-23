import torch 
import math
from torch import nn
from typing import Optional, Union
from utils.tree_old import tree_strtolist, get_unbalanced_leaf_ids, simplify_tree
from utils.awareness import Sign

class IADDT(nn.Module):
    def __init__(self, in_features: int, leaf_width: int, out_features: int, 
                 arch: Optional[Union[str, list]] = None):
        super().__init__()
        self.name = "IADDT"
        self.in_features = in_features
        self.out_features = out_features
        self.leaf_width = leaf_width
        self.activation = nn.ReLU()
        self.sign = Sign.apply
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False) # Awareness parameter

        # Main configurations
        arch = simplify_tree(arch)
        self.arch_list = tree_strtolist(arch)
        leaf_ids, leaf_depths = get_unbalanced_leaf_ids(self.arch_list)
        self.leaf_ids, self.leaf_depths = (torch.tensor(leaf_ids, dtype=torch.long).view(-1, 1), 
                                           torch.tensor(leaf_depths, dtype=torch.long))

        # Unbalanced (real) tree configurations
        self.arch = arch
        self.arch_tensor = torch.tensor([int(i) for i in arch])
        self.n_routers = arch.count('1')
        self.n_leaves = arch.count('2')

        # Balanced tree configurations
        self.depth = len(self.arch_list) - 1
        self.n_leavesb = 2 ** self.depth
        self.n_routersb = 2 ** self.depth - 1

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.router_weights = nn.Parameter(torch.empty((self.n_routersb, in_features)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.router_biases = nn.Parameter(torch.empty((self.n_routersb, 1)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.in_features)
        self.w1s = nn.Parameter(torch.empty((self.n_leavesb, in_features, leaf_width)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leavesb, leaf_width)).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leavesb, leaf_width, out_features)).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leavesb, out_features)).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

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
            if self.alpha < 1.0:
                dec = (self.sign(router_logits).unsqueeze(-1))
                decs = torch.cat((1 - dec, dec), dim=-1)

                dec_mixture_modifier = decs.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)
                dec_mixture = dec_mixture.view(-1, 2 * n_routersb, self.n_leavesb // (2 * n_routersb))
                dec_mixture = dec_mixture * dec_mixture_modifier 
                dec_mixture = dec_mixture.flatten(start_dim=1, end_dim=2)

        # Leave nodes
        outputs = torch.empty((batch_size, self.n_leaves, self.out_features), 
                                 device=device)
        for i in range(self.n_leaves):
            l, d = self.leaf_ids[i], self.leaf_depths[i]
            logits = torch.matmul(x, self.w1s[l].transpose(0, 1).flatten(1, 2))
            logits = logits.view(batch_size, 1, self.leaf_width)
            logits += self.b1s[l].view(1, self.leaf_width)
            leaf_activations = self.activation(logits)
            outputs[:, i] = (torch.matmul(
                leaf_activations,
                self.w2s[l]
            ) + self.b2s[l]).squeeze()

            n_l = 2 ** (self.depth - d) # Number of leaves sharing the same parent

            leaf_prob_mixture = prob_mixture[:, l:(l+n_l)].sum(dim=1, keepdim=True)
            if self.alpha < 1.0:
                leaf_dec_mixture = dec_mixture[:, l:(l+n_l)].sum(dim=1, keepdim=True)
                outputs[:, i] *= (leaf_prob_mixture * self.alpha 
                                  + leaf_dec_mixture * (1-self.alpha))
            else:
                outputs[:, i] *= leaf_prob_mixture
# w1s = self.w1s  # [n_leaves, in_features, leaf_width]
# b1s = self.b1s  # [n_leaves, leaf_width]
# w2s = self.w2s  # [n_leaves, leaf_width, out_features]
# b2s = self.b2s  # [n_leaves, out_features]
#
# # Compute logits for all leaves in parallel
# # x: [batch_size, in_features]
# # w1s: [n_leaves, in_features, leaf_width]
# # Output: [batch_size, n_leaves, leaf_width]
# logits = torch.einsum('bi,nij->bnj', x, w1s) + b1s.unsqueeze(0)  # [batch_size, n_leaves, leaf_width]
#
# # Activation
# leaf_activations = self.activation(logits)  # [batch_size, n_leaves, leaf_width]
#
# # Compute outputs for all leaves in parallel
# # w2s: [n_leaves, leaf_width, out_features]
# # b2s: [n_leaves, out_features]
# outputs = torch.einsum('bnj,njo->bno', leaf_activations, w2s) + b2s.unsqueeze(0)  # [batch_size, n_leaves, out_features]
#
# # If out_features == 1, squeeze last dimension
# outputs = outputs.squeeze(-1)  # [batch_size, n_leaves]
#
# # Compute leaf_prob_mixture for all leaves in parallel
# leaf_prob_mixture = torch.zeros_like(outputs)  # [batch_size, n_leaves]
# for i in range(self.n_leaves):
#     l, d = self.leaf_ids[i], self.leaf_depths[i]
#     n_l = 2 ** (self.depth - d)
#     leaf_prob_mixture[:, i] = prob_mixture[:, l:(l+n_l)].sum(dim=1)

# Multiply outputs by leaf_prob_mixture
# outputs *= leaf_prob_mixture  # [batch_size, n_leaves]
        outputs = outputs.sum(dim=1)
        return outputs

    def eval_forward(self, x: torch.Tensor, leaf_ids: bool = False) -> torch.Tensor:
        batch_size, device = x.shape[0], x.device

        # Router nodes
        routers = torch.zeros((batch_size,), dtype=torch.long, device=device)
        if self.depth == 0: # Edge case: no routers, only one leaf
            breakpoint()
            print(arch)
        for d in range(self.depth):
            w = self.router_weights.index_select(dim=0, index=routers)
            b = self.router_biases.index_select(dim=0, index=routers)
            logits = torch.bmm(x.unsqueeze(1), w.unsqueeze(-1)).squeeze(-1) + b
            choices = (logits.squeeze(-1) >= 0)

            platform, next_platform = 2 ** d - 1, 2 ** (d+1) - 1
            routers = ((routers - platform) * 2  + choices + next_platform)

# w1s_batch = self.w1s[leaf_for_sample]   # [batch_size, in_features, leaf_width]
# b1s_batch = self.b1s[leaf_for_sample]   # [batch_size, leaf_width]
# w2s_batch = self.w2s[leaf_for_sample]   # [batch_size, leaf_width, out_features]
# b2s_batch = self.b2s[leaf_for_sample]   # [batch_size, out_features]
#
# # First layer
# logits = torch.bmm(x.unsqueeze(1), w1s_batch).squeeze(1) + b1s_batch  # [batch_size, leaf_width]
# activations = self.activation(logits)                                 # [batch_size, leaf_width]
#
# # Second layer
# outputs = torch.bmm(activations.unsqueeze(1), w2s_batch).squeeze(1) + b2s_batch  # [batch_size, out_features]
        # Leaf nodes
        leaves = routers - next_platform              # (batch_size,)
        outputs = torch.empty((batch_size, self.out_features), device=device)
        for i in range(self.n_leaves):
            l, d = self.leaf_ids[i], self.leaf_depths[i]
            n_l = 2 ** (self.depth - d) # Number of leaves sharing the same parent
            leaf_indices, = torch.where((leaves >= l) & (leaves < (l+n_l)))

            logits = torch.matmul(x[leaf_indices], self.w1s[l]) + self.b1s[l].unsqueeze(0)                                               # (1, self.leaf_width)
            activations = self.activation(logits)           # (1, self.leaf_width)
            logits = torch.matmul(activations, self.w2s[l]) + self.b2s[l].unsqueeze(0)                                               # (1, self.leaf_width)
            outputs[leaf_indices] = logits

        if leaf_ids:
            return outputs, leaves
        return outputs

    def cal_complexity(self) -> tuple[float, float, float]: # Return #params, min MAC, max MAC
        leaf_params = (self.in_features * self.leaf_width + 
                       self.leaf_width * self.out_features)
        max_macs = self.depth * self.in_features + leaf_params
        min_macs = self.leaf_depths.min() * self.in_features + leaf_params
        n_params = self.n_routers * self.in_features + self.n_leaves * leaf_params
        return n_params, min_macs, max_macs

    def get_config(self):
        n_params, min_macs, max_macs = self.cal_complexity()
        return {
                'model_name': self.name,
                'depth': self.depth,
                'leaf_width': self.leaf_width,
                'n_routers': self.n_routers,
                'n_leaves': self.n_leaves,
                'n_routers': self.n_routers,
                'arch': self.arch,
                'arch list': self.arch_list,
                'in_features': self.in_features,
                'out_features': self.out_features,
                'n_params': n_params,
                'min_macs': min_macs,
                'max_macs': max_macs,
                'leaf_ids': self.leaf_ids.tolist(),
                }

    def get_leaf_stats(self, leaf_ids):
        # Returns how many inputs each leaf received in the current batch
        return torch.tensor([(leaf_ids == i).sum() for i in self.leaf_ids])
    
    def get_avg_macs(self, leaf_stats) -> float:
        leaf_params = (self.in_features * self.leaf_width + 
                       self.leaf_width * self.out_features)
        leaf_depths = torch.log2(self.leaf_ids+1).floor().to(leaf_stats.device)
        avg_macs = ((leaf_stats * leaf_depths.squeeze() * self.in_features).sum() + leaf_params).item()
        return avg_macs

    def get_balance_score(self) -> float:
        # std. of depths
        leaf_depths = torch.log2(self.leaf_ids+1).floor()
        return leaf_depths.std().item()

