from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

DATA_DIR = "data/HAR"
SETS = ["train", "valid", "test"]

class HARLoader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'HAR'
        self.data_dir = Path(DATA_DIR)
        trainset = HARSubset("train")
        testset = HARSubset("test")
        validset = HARSubset("val")

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

        self.in_chan = 1
        self.in_size = (1, 300)
        self.out_dim = 6

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(torch.tensor(img))
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

class HARSubset(Dataset):
    def __init__(self, fold="train"):
        self.data = pickle.load(open(f"{DATA_DIR}/{fold}_data.summary", "rb"), 
                                encoding="latin1")
        self.labels = pickle.load(open(f"{DATA_DIR}/{fold}_labels.summary", "rb"), 
                                  encoding="latin1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        y = np.argmax(y, axis=0)
        return x, y
