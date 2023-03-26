import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

WINDOW_SIZE = 8


class QueryDataset(Dataset):
    def __init__(self, root_dir='data.csv'):
        self.dataset = transform(root_dir)

    def __len__(self):
        return len(self.dataset) - WINDOW_SIZE

    def __getitem__(self, item):
        x = self.dataset.loc[item:item + WINDOW_SIZE - 1]
        feature = torch.tensor(x.values)
        y = self.dataset.loc[item + WINDOW_SIZE, :]
        label = torch.tensor(y.values)
        return feature, label


def transform(root_dir):
    df = pd.read_csv(root_dir, usecols=[1])
    df = df.dropna()
    return df
