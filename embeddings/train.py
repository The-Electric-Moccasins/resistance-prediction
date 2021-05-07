from embeddings.autoencoder import Autoencoder
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, dataset, num_epochs=5, batch_size=64, learning_rate=1e-3):
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
            X = X.to(device)
            recon = model(X.float())
            loss = criterion(recon, X.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        print('.', end="")
        losses.append(float(loss))
#         writer.add_scalar(f'training loss {num_epochs}',
#                             loss,
#                             epoch * len(train_loader))
        # stop training if loss is low
        if loss < 0.0050:
            break
    return outputs, losses