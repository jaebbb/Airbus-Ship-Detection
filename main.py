import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
import datetime
from datasets.dataset import ShipDataset
import matplotlib.pyplot as plt
from configs import config

Dataset_train = ShipDataset()
DataLoader_train = DataLoader(Dataset_train, batch_size=config.train_batchsize, shuffle=True, num_workers=config.train_num_workers)

