import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Siamese(nn.Module):

    def __init__(self, input_size):
        super(Siamese, self).__init__()
        self.input_size = input_size

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 8),
                                  nn.MaxPool2d((2,3)),
                                  nn.Conv2d(32, 16, 5),
                                  nn.MaxPool2d((2,3)),
                                  nn.Conv2d(16, 8, 3),
                                  nn.MaxPool2d((2,3)),
                                  nn.Flatten())

        self.fc_size = self.conv(torch.zeros((3, input_size[0], input_size[1])).unsqueeze(0)).shape[-1]

        self.fc = nn.Sequential(nn.Linear(self.fc_size, 128),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.Dropout(0.1),
                                nn.ReLU())

        self.out = nn.Sequential(nn.Linear(128, 128),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(128, 2))

    def forward_one(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        y1 = self.forward_one(x1)
        y2 = self.forward_one(x2)
        diff = torch.abs(y1 - y2)
        out = self.out(diff)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    model = Siamese((400,600))
    print(model)
