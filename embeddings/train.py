from embeddings.autoencoder import Autoencoder
import numpy as np
from numpy import random


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt



def add_noise(X, p:float, device):
    X = X * 2 - 1
    #noise = random.binomial(1, p, size=X.numel()).reshape(X.shape).to_device(device)*2 - 1
    m = torch.distributions.binomial.Binomial(1, p)
    noise = m.sample(sample_shape=X.shape).to(device) *2 - 1
    
    #print(noise)
    noised = X * noise
    noised = (noised + 1 ) / 2
    return noised

def train(model, dataset, num_epochs=5, batch_size=64, learning_rate=1e-3, denoising=False, denoise_p=0.1):
    print(f"Will now train AutoEncoder with max {num_epochs} epochs")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
#     criterion = nn.L1Loss()
#     criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    outputs = []
    losses = []
    for epoch in range(num_epochs):
        for X, y in train_loader:
            # Train each autoencoder individually
            X_true = X.detach().to(device)   
            
            # Add noise, but use the original lossless input as the target.
            if denoising:
                X = X.to(device)
#                 X = X.detach().to(device)
                
                X = add_noise(X, denoise_p, device)
            else:
                X = X.to(device)
            recon = model(X.float())
            loss = criterion(recon, X_true.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        print(".", end="")
        if epoch % 10 == 0:
            print('Loss:{:.4f}'.format(float(loss)), end="")
        losses.append(float(loss))
#         writer.add_scalar(f'training loss {num_epochs}',
#                             loss,
#                             epoch * len(train_loader))
        # stop training if loss is low
        if loss < 0.0050:
            break
    return outputs, losses


def plot_loss(losses):
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(losses)