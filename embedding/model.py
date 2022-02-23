import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 16, (2,3), stride = (2,3)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(16, 32, (5,5), stride = (5,5)),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, (2,2), stride = (2,2)),
                                  nn.ReLU(),
                                  )

        self.fc_size = self.conv(torch.zeros((3, 200, 300)).unsqueeze(0)).shape[-1]

    def forward(self, x):
        x = self.conv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 32, (2,2), stride=(2,2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 32, (5,5), stride = (5,5)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 32, (2,2), stride = (2,2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 3, (2,3), stride = (2,3)),
                                    nn.ReLU(),
                                    )

    def forward(self, x):
        x = self.deconv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    test = torch.zeros((1,3,200,300))
    print(enc(test).shape)
    print(dec(enc(test)).shape)
