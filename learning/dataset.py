import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import random

class ImageDataset(Dataset):

    def __init__(self, path = "../res/train", img_size = (400,600), device = torch.device("cuda")):
        self.device = device
        self.path = path
        self.img_size = img_size
        self.data = datasets.ImageFolder(root = self.path,
                                         transform = transforms.Compose([
                                             transforms.Resize(self.img_size),
                                             transforms.ToTensor(),
                                             transforms.ConvertImageDtype(torch.float)]))
        self.num_classes = len(self.data.classes)
        self.targets = torch.tensor(self.data.targets)

    def __getitem__(self, index):
        if index % 2 == 0:
            [i1] = [i2] = random.sample(range(self.num_classes), 1)
        else:
            [i1, i2] = random.sample(range(self.num_classes), 2)
        x1 = random.choice(torch.nonzero(self.targets == i1, as_tuple=False).tolist())[0]
        x2 = random.choice(torch.nonzero(self.targets == i2, as_tuple=False).tolist())[0]
        return self.data[x1][0].to(self.device), self.data[x2][0].to(self.device), torch.tensor(int(i1 == i2), dtype=torch.long, device = self.device)

    def __len__(self):
        return 5000

if __name__ == '__main__':
    dataset = ImageDataset()
    x1, x2, y1 = dataset[0]
    x3, x4, y2 = dataset[1]

    img1 = x1.squeeze().permute(1,2,0)
    img2 = x2.squeeze().permute(1,2,0)
    img3 = x3.squeeze().permute(1,2,0)
    img4 = x4.squeeze().permute(1,2,0)
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img1)
    axarr[0,1].imshow(img2)
    axarr[1,0].imshow(img3)
    axarr[1,1].imshow(img4)
    plt.show()
