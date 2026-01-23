import torch
import torch.nn as nn
import torch.nn.functional as F
from .ff import FF

# Define the Gating Network class
class LinearGating(nn.Module):
    def __init__(self, in_features, num_experts):
        super(LinearGating, self).__init__()
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        x = x.view(len(x), -1)
        return torch.softmax(self.gate(x), dim=1)

# Define the Mixture of Experts Layer class
class MoE(nn.Module):
    def __init__(self, in_features: int, width: int, out_features: int, num_experts: int, 
                 topk: int, mode: str):
        super(MoE, self).__init__()
        self.name = "MoE"
        self.experts = nn.ModuleList(
                [FF(in_features, width, out_features) 
                 for _ in range(num_experts)]
                )
        self.topk = topk
        self.gate = LinearGating(in_features, num_experts)
        self.width = width
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.eval_topk = 1 if mode == "sparse" else self.topk

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.training_forward(x)
        else:
            return self.eval_forward(x, topk=self.eval_topk)

    def training_forward(self, x: torch.Tensor):
        if not self.topk:
            self.topk = self.num_experts
        gating_scores = self.gate(x)
        top_gating_scores, top_indices = gating_scores.topk(self.topk, dim=1, sorted=False)

        mask = torch.zeros_like(gating_scores).scatter_(1, top_indices, 1)
        gating_scores = gating_scores * mask
        gating_scores = F.normalize(gating_scores, p=1, dim=1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', gating_scores, expert_outputs)
        return output

    def eval_forward(self, x: torch.Tensor, topk: int = 1):
        gating_scores = self.gate(x)
        top_gating_scores, top_indices = gating_scores.topk(topk, dim=1, sorted=False)

        mask = torch.zeros_like(gating_scores).scatter_(1, top_indices, 1)
        gating_scores = gating_scores * mask
        gating_scores = F.normalize(gating_scores, p=1, dim=1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', gating_scores, expert_outputs)
        return output

    def cal_complexity(self) -> tuple[float, float]: # Return #params, MAC
        expert_params = (self.in_features * self.width + self.width * self.out_features)
        macs = self.in_features + expert_params * self.topk
        n_params = self.in_features + self.num_experts * expert_params
        return n_params, macs

    def get_config(self):
        n_params, macs = self.cal_complexity()
        return {
                "model_name": self.name,
                "width": self.width,
                "gate": "linear",
                "num_experts": self.num_experts,
                'macs': macs,
                'n_params': n_params,
                }
