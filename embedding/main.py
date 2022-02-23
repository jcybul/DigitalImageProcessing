import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from model import Encoder, Decoder
from dataset import ImageDataset, BasicImageDataset
from learning import train, create_embedding, compute_similar_images

device = torch.device("cuda")
img_size = (200,300)
batch_size = 10
num_epochs = 1

train_data = ImageDataset("../res/train", img_size, device = device)
test_data = BasicImageDataset("../res/test", img_size)

trainloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
testloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

# model = train(encoder, decoder, trainloader, device, num_epochs)

embedding = create_embedding(encoder, testloader, (1,64,5,5), device)

img = test_data[0][0]


idx = compute_similar_images(img, 3, encoder, embedding, device)
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(img)
axarr[0,1].imshow(img)
axarr[1,0].imshow(img)
axarr[1,1].imshow(img)
plt.show()
