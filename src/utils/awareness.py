import torch

class LinearWarmupSched: # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, warmup_rate: float, 
                 warmup: object, epochs: int):
        self.alpha = alpha
        self.epochs = epochs
        self.warmup = warmup
        self.warmup_rate = warmup_rate
        self.warmup_epochs = warmup_rate * epochs
        self.device = self.alpha.device
        self.epoch = 0

    def step(self):
        alpha = self.warmup()
        if self.epoch >= self.warmup_epochs:
            alpha = 1 - (self.epoch+1) / self.epochs
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device).clamp(max=100.0))
        self.epoch += 1

    def get_base_config(self):
        return {
                'asig_warmupr': self.warmup_rate,
                }

    def get_config(self):
        return self.get_base_config()

class StepSched:
    def __init__(self, alpha: torch.nn.Parameter, steps:int, epochs: int):
        self.alpha = alpha
        self.epochs = epochs
        self.steps = steps
        self.device = self.alpha.device
        self.epoch = 0

    def step(self):
        # TODO
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device).clamp(max=100.0))
        self.epoch += 1

    def get_base_config(self):
        return {
                'a_steps': self.steps,
                }

    def get_config(self):
        return self.get_base_config()

class NoSched:
    def __init__(self, alpha: torch.nn.Parameter):
        return

    def step(self):
        return

    def get_config(self):
        return {}

class WarmUp:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epoch = 0
    
    def __call__(self):
        return 1.0

class Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (torch.sign(x) + 1) / 2  # Forward pass

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x.abs()

