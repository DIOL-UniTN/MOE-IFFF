import torch
from tqdm import tqdm

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch(model, optim, loader, criterion, epoch, device, tqdm_disable: bool = False):
    model.train()
    correct, running_loss = 0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader), disable=tqdm_disable):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # back propagation
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    correct, running_loss = 0, 0.0
    sample_leaves, sample_preds = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset), 

@torch.no_grad()
def eval_ddt(model, loader, criterion, device):
    model.eval()
    correct, running_loss = 0, 0.0
    sample_leaves, sample_preds = [], []
    leaf_stats = torch.zeros((model.n_leaves,))
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, leaf_ids = model(inputs, leaf_ids=True)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()
        leaf_stats += model.get_leaf_stats(leaf_ids)

    return (running_loss/len(loader), correct/len(loader.dataset), 
            leaf_stats / leaf_stats.sum())
