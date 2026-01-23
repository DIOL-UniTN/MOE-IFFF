from pathlib import Path
import logging
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

DATA_DIR = "data/cifar-10-batches-py"
SETS = ["train", "valid", "test"]

class CIFAR10Loader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'CIFAR10'
        self.data_dir = Path(DATA_DIR)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        whole_tset = CIFAR10(root=self.data_dir.parent, train=True, download=True,
                             transform=transform)
        testset = CIFAR10(root=self.data_dir.parent, train=False, download=True, 
                          transform=transform)

        if not (self.data_dir/"perm.pt").exists(): # For a fixed valid set every run (reporducibility)
            logging.warning(
                    f"The file {self.data_dir/'perm.pt'} doesn't exist. "
                    "It will be generated now. Yet, the results of our experiments "
                    "may not be reproduced. Please check the ``root`` path or run " 
                    "`dvc pull` to download it"
                    )
            perm = torch.randperm(len(whole_tset))
            torch.save(perm, self.data_dir/"perm.pt")
        perm = torch.load(self.data_dir/"perm.pt")

        val_len = int(len(perm)*0.1) # 10% for validation

        trainset = Subset(whole_tset, perm[val_len:])
        validset = Subset(whole_tset, perm[:val_len])

        tensor_missing = sum([(not (self.data_dir/f"{dset}_tensors.pt").exists() 
                               or not (self.data_dir/f"{dset}_labels.pt").exists())
                              for dset in SETS])
        if tensor_missing:
               self.get_tensors([trainset, validset, testset])

        train_tensors, train_labels = (torch.load(self.data_dir/"train_tensors.pt"), 
                                       torch.load(self.data_dir/"train_labels.pt"))
        valid_tensors, valid_labels = (torch.load(self.data_dir/"valid_tensors.pt"), 
                                       torch.load(self.data_dir/"valid_labels.pt"))
        test_tensors, test_labels = (torch.load(self.data_dir/"test_tensors.pt"), 
                                       torch.load(self.data_dir/"test_labels.pt"))

        trains = TensorDataset(train_tensors, train_labels)
        valids = TensorDataset(valid_tensors, valid_labels)
        tests = TensorDataset(test_tensors, test_labels)

        self.train = DataLoader(trains, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(valids, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(tests, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 3
        self.in_size = (32, 32)
        self.out_dim = 10

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

    def get_config(self):
        return {
                "task": self.name,
                "in_chan": self.in_chan,
                "in_size": self.in_size,
                "out_dim": self.out_dim,
                "train_samples": len(self.train.dataset),
                "valid_samples": len(self.valid.dataset),
                "test_samples": len(self.test.dataset),
                }
