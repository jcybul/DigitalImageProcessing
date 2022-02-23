import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

class ImageDataset(Dataset):

    def __init__(self, path = "../res/test", img_size = (200,300), device = torch.device("cuda")):
        self.device = device
        self.path = path
        self.img_size = img_size
        self.data = datasets.ImageFolder(root = self.path,
                                         transform = transforms.Compose([
                                             transforms.Resize(self.img_size),
                                             transforms.ToTensor(),
                                             transforms.ConvertImageDtype(torch.float),
                                             ]))
        self.num_classes = len(self.data.classes)
        self.targets = torch.tensor(self.data.targets)

    def sample_target(self, target):
        return self.data[random.choice(torch.nonzero(self.targets == target, as_tuple=False).tolist())[0]][0].to(self.device)

    def __getitem__(self, index):
        [i] = random.sample(range(self.num_classes), 1)
        x1 = self.sample_target(i)
        x2 = self.sample_target(i)
        return x1, torch.mean(x1), x2, torch.mean(x2)

    def __len__(self):
        return 1000

def BasicImageDataset(path, img_size):
    return datasets.ImageFolder(root = path,
                                transform = transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float),
                                    ]))


if __name__ == '__main__':
    dataset = BasicImageDataset('../res/test', (200,300))
    x1 = dataset[0][0]
    print(x1.shape)
