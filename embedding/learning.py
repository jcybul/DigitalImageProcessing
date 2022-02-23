import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def train(encoder, decoder, trainloader, device, epochs):
    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(model.parameters(), lr = 5e-4, momentum=0.9)
    max_acc = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        bar = tqdm(enumerate(trainloader, 0), total = len(trainloader))
        for i, (x1, m1, x2, m2) in bar:
            X1, X2 = Variable(x1), Variable(x2)

            optimizer.zero_grad()

            feature = encoder(X1)
            generated = decoder(feature)

            loss = criterion(generated, X2)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            bar.set_description('Epoch %d, loss: %.3f'%(epoch + 1, running_loss))
    return running_loss

def create_embedding(encoder, dataloader, embedding_dim, device):
    encoder.eval()
    embedding = torch.randn(embedding_dim)

    with torch.no_grad():
        for i, (x1, x2) in enumerate(dataloader):
            x1 = x1.to(device)
            enc_output = encoder(x1).cpu()
            embedding = torch.cat((embedding, enc_output), 0)
    return embedding

def compute_similar_images(image, num_images, encoder, embedding, device):
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = encoder(image).cpu().detach().numpy()

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    return indices_list
