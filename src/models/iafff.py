import torch 
import math
from torch import nn
from typing import Optional
from utils.awareness import Sign

class IAFFF(nn.Module):
    def __init__(self, in_features: int, leaf_width: int, out_features: int, depth: int):
        super().__init__()
        self.name = "IAFFF"
        self.in_features = in_features
        self.leaf_width = leaf_width
        self.depth = depth
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.sign = Sign.apply
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False) # Awareness parameter

        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.node_weights = nn.Parameter(torch.empty((self.n_nodes, in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.in_features)
        self.w1s = nn.Parameter(torch.empty((self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def forward(self, x: torch.Tensor, leaf_ids:bool=False):
        x = x.view(len(x), -1)
        if self.training:
            return self.training_forward(x)
        else:
            return self.eval_forward(x, leaf_ids)

    def training_forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        prob_mixture = torch.ones((batch_size, self.n_leaves), device=x.device)
        dec_mixture = torch.ones((batch_size, self.n_leaves), device=x.device)
        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)

            prob = torch.sigmoid(boundary_plane_logits)                              # (batch_size, n_nodes)
            not_prob = 1 - prob                                   # (batch_size, n_nodes)

            prob_mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (not_prob.unsqueeze(-1), prob.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            prob_mixture = prob_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            prob_mixture.mul_(prob_mixture_modifier)                                                          # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            prob_mixture = prob_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

            # Decs.
            if self.alpha < 1.0:
                dec = self.sign(boundary_plane_logits)                              # (batch_size, n_nodes)
                not_dec = 1 - dec                                   # (batch_size, n_nodes)

                dec_mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                    (not_dec.unsqueeze(-1), dec.unsqueeze(-1)),
                    dim=-1
                ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
                dec_mixture = dec_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
                dec_mixture.mul_(dec_mixture_modifier)                                                          # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
                dec_mixture = dec_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

        element_logits = torch.matmul(x, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s.view(1, *self.b1s.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)

        # new_logits = torch.einsum('bnd,ndh->bnh', element_activations, self.w2s) + self.b2s
        new_logits = torch.empty((batch_size, self.n_leaves, self.out_features), dtype=torch.float, device=x.device)
        for i in range(self.n_leaves):
            new_logits[:, i] = torch.matmul(
                element_activations[:, i],
                self.w2s[i]
            ) + self.b2s[i]
        # new_logits has shape (batch_size, self.n_leaves, self.out_features)

        prob_logits = new_logits * prob_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
        if self.alpha < 1.0:
            dec_logits = new_logits * dec_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
            logits = (prob_logits.sum(dim=1) * self.alpha 
                      + dec_logits.sum(dim=1) * (1-self.alpha))
        else:
            logits = prob_logits.sum(dim=1)
        return logits

    def eval_forward(self, x: torch.Tensor, leaf_ids: bool = False) -> torch.Tensor:
        batch_size = x.shape[0]

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)       # (batch_size, in_features)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)       # (batch_size, 1)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))       # (batch_size, 1, 1)
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets                     # (batch_size, 1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)          # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device) # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform  # (batch_size,)

        leaves = current_nodes - next_platform              # (batch_size,)

        # w1s_batch = self.w1s[leaves]   # [batch_size, in_features, leaf_width]
        # b1s_batch = self.b1s[leaves]   # [batch_size, leaf_width]
        # w2s_batch = self.w2s[leaves]   # [batch_size, leaf_width, out_features]
        # logits = torch.bmm(x.unsqueeze(1), w1s_batch).squeeze(1)  # [batch_size, leaf_width]
        # logits += b1s_batch                                       # [batch_size, leaf_width]
        # activations = self.activation(logits)                     # [batch_size, leaf_width]
        # new_logits = torch.bmm(activations.unsqueeze(1), w2s_batch).squeeze(1)  # [batch_size, out_features]

        new_logits = torch.empty((batch_size, self.out_features), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            logits = torch.matmul(
                x[i].unsqueeze(0),                  # (1, self.in_features)
                self.w1s[leaf_index]                # (self.in_features, self.leaf_width)
            )                                               # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)           # (1, self.leaf_width)
            new_logits[i] = torch.matmul(
                activations,
                self.w2s[leaf_index]
            ).squeeze(-2)                                   # (1, self.out_features)

        if leaf_ids:
            return new_logits, leaves
        return new_logits

    def cal_complexity(self) -> tuple[float, float]: # Return #params, MAC
        leaf_params = (self.in_features * self.leaf_width + 
                       self.leaf_width * self.out_features)
        macs = self.depth * self.in_features + leaf_params
        n_params = self.n_nodes * self.in_features + self.n_leaves * leaf_params
        return n_params, macs

    def get_config(self):
        n_params, macs = self.cal_complexity()
        return {
                'depth': self.depth,
                'leaf_width': self.leaf_width,
                'in_features': self.in_features,
                'out_features': self.out_features,
                'macs': macs,
                'n_params': n_params,
                'model_name': self.name,
                }

    def get_leaf_stats(self, leaf_ids):
         # Returns how many inputs each leaf received in the current batch
         return torch.tensor([(leaf_ids == i).sum() for i in range(self.n_leaves)])
