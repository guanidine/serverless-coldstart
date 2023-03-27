import torch
from torch import nn, optim

import config


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model: nn.Module, optimizer: optim.Optimizer, lr: float):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> Loading success")


def normalize(data):
    _range = config.FUNCTION_NUM / 2
    return (data - _range) / _range


def denormalize(data):
    _range = config.FUNCTION_NUM / 2
    return data * _range + _range


def cal_reward(query, action):
    query += 1
    action += 1
    if query > action * config.UPPER_SLA or query < action * config.LOWER_SLA:
        return -1
    else:
        return (
                min(query / (action + 1e-3), 1)
                - max(query - action, 0) / (query + 1e-3)
        )
