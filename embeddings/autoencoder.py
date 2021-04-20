
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=256 , bias=True),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=128, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=num_features, bias=True ),
            #nn.Sigmoid()
            nn.Tanh()
#             nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

