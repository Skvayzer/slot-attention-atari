from torch import nn
import torch

class Encoder(nn.Module):
    """
    CNN encoder for set prediction task, which reduces the spatial resolution 4 times
    """
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, hidden_size, 5, padding=(2, 2)), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.ZeroPad2d((1, 3, 1, 3)),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(0, 0), stride=2), nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 5, padding=(2, 2)), nn.ReLU()
        ])

    def forward(self, inputs):
        return self.layers(inputs)

class WaymoEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.resnet34 = torch.hub.load("pytorch/vision", "resnet34", weights="IMAGENET1K_V1")
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, hidden_size*16*24)


    def forward(self, inputs):
        x = self.resnet34(inputs)
        x = x.reshape(-1, self.hidden_size, 16, 24)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x