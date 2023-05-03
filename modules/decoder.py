from torch import nn


class Decoder(nn.Module):
    """
    Decoder for autoencoder model
    """
    def __init__(self, num_channels=64):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(1,1), output_padding=0, stride=2),
        ])
        self.final_module = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, 5, padding=(2, 2), output_padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels, 4, 3, padding=(1,1), output_padding=0, stride=1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)[:, :, :-1, :-1]
        return self.final_module(x)


class MultiDspritesDecoder(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int = 64,
                 hidden_channels: int = 64,
                 out_channels: int = 4,
                 mode='clevr_with_masks'):
        super(MultiDspritesDecoder, self).__init__()
        if mode in ['clevr_with_masks']:
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(1, 1), padding=2), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, out_channels,
                                   kernel_size=3, stride=(1, 1), padding=1)
            )
        elif mode == 'multi_dsprites' or mode == 'tetrominoes':
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, hidden_channels,
                                   kernel_size=3, stride=(1, 1), padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=3, stride=(1, 1), padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=3, stride=(1, 1), padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, out_channels,
                                   kernel_size=3, stride=(1, 1), padding=1), nn.ReLU(),
            )

        else:
            raise ValueError("Mode should be either of ['clevr_with_masks', 'multi_dsprites', 'tetrominoes'")

    def forward(self, x):
        return self.decoder_cnn(x)

class TetrominoesDecoder(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int = 35*35,
                 hidden_channels: int = 256,
                 out_channels: int = 4):
        super(TetrominoesDecoder, self).__init__()

        self.decoder_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, out_channels), nn.ReLU(),

        )

    def forward(self, x):
        return self.decoder_mlp(x)


class WaymoDecoder(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int = 64,
                 hidden_channels: int = 64,
                 out_channels: int = 4
                 ):
        super(WaymoDecoder, self).__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels,
                               kernel_size=5, stride=(2, 2), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=(2, 2), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=(2, 2), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=(1, 1), padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=5, stride=(1, 1), padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels,
                               kernel_size=6, stride=(1, 1), padding=1)
        )


    def forward(self, x):
        return self.decoder_cnn(x)

