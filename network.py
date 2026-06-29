import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim):
        super(Network, self).__init__()
        self.views = views
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(views)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim) for v in range(views)])

    def forward(self, xs):
        zs, xrs = [], []
        for v in range(self.views):
            z = self.encoders[v](xs[v])
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return zs, xrs
