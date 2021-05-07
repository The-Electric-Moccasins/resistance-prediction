
# d= 128
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_features, dropout_p=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=num_features // 2 , bias=True),
            nn.Dropout(p = dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=num_features // 2, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=256, out_features=num_features // 2, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=num_features // 2, out_features=num_features, bias=True ),
            #nn.Sigmoid()
            nn.ReLU()
#             nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
