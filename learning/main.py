import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

from model import Siamese
from dataset import ImageDataset
from learning import train, test

device = torch.device("cuda")
img_size = (100,150)
batch_size = 10
num_epochs = 10

train_data = ImageDataset("../res/train", img_size, device = device)
test_data = ImageDataset("../res/test", img_size, device = device)

trainloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
testloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

model = Siamese(input_size = img_size).to(device)

model = train(model, trainloader, testloader, device, num_epochs)

model.save('model.pth')
