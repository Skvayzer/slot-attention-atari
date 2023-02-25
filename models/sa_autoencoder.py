import os
import sys

import pytorch_lightning as pl
import torch
import wandb

from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, CoordQuantizer
from modules.slot_attention import SlotAttentionBase
from utils import spatial_broadcast, spatial_flatten, adjusted_rand_index


class SlotAttentionAE(nn.Module):
    """
    Slot attention based autoencoder for object discovery task
    """

    def __init__(self,
                 resolution=(105, 80),
                 num_slots=6,
                 num_iters=3,
                 in_channels=3,
                 slot_size=64,
                 log=None,
                 hidden_size=64,
                 dataset='',
                 task='',
                 quantization=False,
                 nums=[8, 8, 8, 8],
                 beta=1,
                 lr=4e-4,
                 num_steps=int(3e5), **kwargs
                 ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.task = task
        self.log = log
        self.quantization = quantization

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(3)]
        )
        self.decoder_initial_size = (8, 8)

        # Decoder
        self.decoder = Decoder()

        self.enc_emb = PosEmbeds(64, self.resolution)
        self.dec_emb = PosEmbeds(64, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                hidden_dim=slot_size * 2)
        if self.quantization:
            self.slots_lin = nn.Linear(16 * len(nums) + 64, hidden_size)
            self.coord_quantizer = CoordQuantizer(nums)
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta


    def forward(self, inputs, test=False):
        x = self.encoder(inputs)
        x = self.enc_emb(x)

        x = spatial_flatten(x[0])
        x = self.layer_norm(x)
        x = self.mlp(x)

        slots = self.slot_attention(x)

        kl_loss = 0
        if self.quantization:
            props, coords, kl_loss = self.coord_quantizer(slots, test)
            # print("\n\nATTENTION! props/coords : ", props.shape, coords.shape, file=sys.stderr, flush=True)

            slots = torch.cat([props, coords], dim=-1)
            slots = self.slots_lin(slots)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        x = self.dec_emb(x)
        x = self.decoder(x[0])

        x = x.reshape(inputs.shape[0], self.num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons, kl_loss, masks

    def step(self, batch):
        result, _, kl_loss, _ = self.forward(batch)
        print("result shape", result.shape, file=sys.stderr, flush=True)
        print("batch shape", batch.shape, file=sys.stderr, flush=True)

        loss = F.mse_loss(result, batch)
        return loss, kl_loss

    def training_step(self, batch, optimizer):
        loss, kl_loss = self.step(batch)
        wandb.log({'Training MSE': loss})
        if self.quantization:
            wandb.log({'Training KL': kl_loss})

        loss = loss #+ kl_loss * self.beta

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    def validation_step(self, batch):
        loss, kl_loss = self.step(batch)
        wandb.log({'Validation MSE': loss})
        if self.quantization:
            wandb.log({'Validation KL': kl_loss})

        imgs = batch

        result, recons, _, pred_masks = self.forward(imgs)
        # print("\n\nATTENTION! imgs: ", imgs.shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! recons: ", recons.shape, file=sys.stderr, flush=True)

        wandb.log({
            'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
            'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)]
        })

        wandb.log({
            f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]
            for i in range(self.num_slots)
        })

        return loss

    def validation_epoch_end(self):
        if self.current_epoch % 5 == 0:
            save_path = "./sa_autoencoder_end_to_end/" + f'{self.dataset}' + '/' + f'{self.task}'
            self.trainer.save_checkpoint(os.path.join(save_path, f"{self.current_epoch}_{self.beta}_{self.task}_{self.dataset}_od_pretrained.ckpt"))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=int(self.num_steps), pct_start=0.05)
        return [optimizer], [scheduler]
