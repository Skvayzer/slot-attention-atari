import os
import sys

import pytorch_lightning as pl
import torch
import wandb

from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, ISAPosEmbeds, CoordQuantizer, MultiDspritesDecoder
from modules.slot_attention import SlotAttentionBase, InvariantSlotAttention
from utils import spatial_broadcast, spatial_flatten, adjusted_rand_index, mask_iou


class InvariantSlotAttentionAE(pl.LightningModule):
    """
    Slot attention based autoencoder for object discovery task
    """

    def __init__(self,
                 resolution=(128, 128),
                 num_slots=15,
                 val_num_slots=15,
                 num_iters=3,
                 in_channels=3,
                 slot_size=32,
                 hidden_size=32,
                 dataset='',
                 task='',
                 invariance=True,
                 beta=0.01,
                 lr=4e-4,
                 num_steps=int(3e5),
                 train_dataloader=None, **kwargs
                 ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.val_num_slots = val_num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.task = task
        self.invariance = invariance
        self.train_dataloader = train_dataloader

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(3)]
        )


        # Decoder
        if dataset=='seaquest':
            self.decoder_initial_size = (8, 8)
            self.decoder = Decoder(num_channels=hidden_size)
        else:
            self.decoder_initial_size = self.resolution
            self.decoder = MultiDspritesDecoder(in_channels=self.slot_size,
                                   hidden_channels=self.hidden_size,
                                   out_channels=4,
                                   mode=dataset)

        # self.enc_emb = ISAPosEmbeds(hidden_size, self.resolution)
        self.dec_emb = ISAPosEmbeds(hidden_size, self.decoder_initial_size)
        self.h = nn.Linear(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        if invariance:
            self.slot_attention = InvariantSlotAttention(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                hidden_dim=slot_size * 2, resolution=resolution, enc_hidden_size=hidden_size)
        else:
            self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size, resolution=resolution,
                                                         hidden_dim=slot_size * 2)
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta
        self.save_hyperparameters()


    def forward(self, inputs, num_slots=None, test=False):
        x = self.encoder(inputs)
        # print(f"\n\nATTENTION! encoded {encoded.shape} ", file=sys.stderr, flush=True)
        torch.autograd.set_detect_anomaly(True)

        if not self.invariance:
            x = self.enc_emb(x)
            # print(f"\n\nATTENTION! x {x[0].shape} {x[1]} ", file=sys.stderr, flush=True)
            x = spatial_flatten(x[0])
            # print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)

            x = self.layer_norm(x)
            # print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)

            x = self.mlp(x)
            # print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)




        # print(f"\n\nATTENTION! num slots: {num_slots} ", file=sys.stderr, flush=True)
        if num_slots is None:
            num_slots = self.num_slots

        slots, S_p, S_r = self.slot_attention(x, n_s=num_slots)
        S_p = S_p.unsqueeze(dim=2)
        print(f"\n\nATTENTION! before dec S_p: {S_p.shape} ", file=sys.stderr, flush=True)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        # _, pos_emb = self.dec_emb(x)
        grid = self.dec_emb.grid.unsqueeze(dim=0).view(1, 1, -1, 2)
        print(f"\n\nATTENTION! before dec pos emb: {grid.shape} ", file=sys.stderr, flush=True)
        print(f"\n\nATTENTION! S_r: {S_r.shape} ", file=sys.stderr, flush=True)

        rel_grid = grid - S_p
        print(f"\n\nATTENTION! rel_grid: {rel_grid.shape} ", file=sys.stderr, flush=True)

        # rel_grid_final = torch.zeros(rel_grid.shape).cuda()
        # rel_grid = torch.einsum('bskd,bsijd->bsijk', torch.inverse(S_r), grid - S_p)
        S_r_inverse = torch.inverse(S_r)
        print(f"\n\nATTENTION! S_r_inv: {S_r_inverse.shape} ", file=sys.stderr, flush=True)
        rel_grid_final = torch.einsum("bsij,bskj->bski", S_r_inverse, rel_grid)
        # for b in range(S_p.shape[0]):
        #     for s in range(num_slots):
        #         rel_grid_final[b, s, :, :] = (S_r_inverse[b, s, :, :] @ rel_grid[b, s, :, :].T).T
        print(f"\n\nATTENTION! before dec: {x.shape} ", file=sys.stderr, flush=True)
        print(f"\n\nATTENTION! self.h(rel_grid): {self.h(rel_grid_final).reshape(-1, self.hidden_size, *self.decoder_initial_size).shape} ", file=sys.stderr, flush=True)

        x = self.decoder(x + self.h(rel_grid_final).reshape(-1, self.hidden_size, *self.decoder_initial_size))

        x = x.reshape(inputs.shape[0], num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        # print(f"\n\nATTENTION! masks: {masks}, mask shape: {masks.shape} ", file=sys.stderr, flush=True)
        iou_loss = 0
        if self.beta != 0:
            iou_loss = mask_iou(masks)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons, iou_loss, masks

    def step(self, batch, num_slots=None):
        if self.dataset == "celeba":
            imgs = batch[0]
        else:
            imgs = batch['image']
        result, _, iou_loss, _ = self(imgs, num_slots)
        loss = F.mse_loss(result, imgs)
        return loss, iou_loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, iou_loss = self.step(batch)
        self.log('Training MSE', loss)
        self.log('Training iou loss', iou_loss)


        loss = loss + iou_loss * self.beta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log('lr', sch.get_last_lr()[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou_loss = self.step(batch, num_slots=self.val_num_slots)
        self.log('Validation MSE', loss)
        self.log('Validation iou', iou_loss)

        if batch_idx == 0:
            imgs = batch['image']
            imgs = imgs[:8]

            result, recons, _, pred_masks = self(imgs, num_slots=self.val_num_slots)
            print("\n\nATTENTION! imgs: ", imgs.shape, file=sys.stderr, flush=True)
            print("\n\nATTENTION! recons: ", recons.shape, file=sys.stderr, flush=True)

            self.trainer.logger.experiment.log({
                'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
                'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)]
            })

            for i in range(self.val_num_slots):
                # print(f"\n\n\nATTENTION! {i} slot: ", recons[:, i], file=sys.stderr, flush=True)
                self.trainer.logger.experiment.log({
                    f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]

                })

        return loss

    def validation_epoch_end(self, outputdata):
        if self.current_epoch % 5 == 0:
            save_path = "./sa_autoencoder_end_to_end/" + f'{self.dataset}' + '/' + f'{self.task}'
            self.trainer.save_checkpoint(os.path.join(save_path, f"{self.current_epoch}_{self.beta}_{self.task}_{self.dataset}_od_pretrained.ckpt"))

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=int(self.num_steps), pct_start=0.05)
    #     return [optimizer], [scheduler]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        warmup_steps_pct = 0.02
        decay_steps_pct = 0.2
        scheduler_gamma = 0.5
        max_epochs = 100
        total_steps = max_epochs * len(self.train_dataloader)

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )


class SlotAttentionAE(pl.LightningModule):
    """
    Slot attention based autoencoder for object discovery task
    """

    def __init__(self,
                 resolution=(128, 128),
                 num_slots=10,
                 val_num_slots=20,
                 num_iters=3,
                 in_channels=3,
                 slot_size=64,
                 hidden_size=32,
                 dataset='',
                 task='',
                 invariance=False,
                 beta=0.01,
                 lr=4e-4,
                 num_steps=int(3e5),
                 train_dataloader=None, **kwargs

                 ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.val_num_slots = val_num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.task = task
        self.invariance = invariance
        self.train_dataloader = train_dataloader

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(3)]
        )
        self.decoder_initial_size = (8, 8)

        # Decoder
        if dataset == 'seaquest':
            self.decoder_initial_size = (8, 8)
            self.decoder = Decoder(num_channels=hidden_size)
        else:
            self.decoder_initial_size = self.resolution
            self.decoder = MultiDspritesDecoder(in_channels=self.slot_size,
                                                hidden_channels=self.hidden_size,
                                                out_channels=4,
                                                mode=dataset)

        self.enc_emb = PosEmbeds(hidden_size, self.resolution)
        self.dec_emb = PosEmbeds(hidden_size, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        if invariance:
            self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                         hidden_dim=slot_size * 2, enc_hidden_size=hidden_size)
        else:
            self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                    hidden_dim=slot_size * 2)
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta
        self.save_hyperparameters()

    def forward(self, inputs, num_slots=None, test=False):
        x = self.encoder(inputs)
        print(f"\n\nATTENTION! encoded {x.shape} ", file=sys.stderr, flush=True)


        x = self.enc_emb(x)
        print(f"\n\nATTENTION! x {x[0].shape} {x[1].shape} ", file=sys.stderr, flush=True)
        x = spatial_flatten(x[0])
        print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)

        x = self.layer_norm(x)
        print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)

        x = self.mlp(x)
        print(f"\n\nATTENTION! x {x.shape} ", file=sys.stderr, flush=True)

        # print(f"\n\nATTENTION! num slots: {num_slots} ", file=sys.stderr, flush=True)
        if num_slots is None:
            num_slots = self.num_slots

        slots = self.slot_attention(x, n_s=num_slots)
        print(f"\n\nATTENTION! slots {slots.shape} ", file=sys.stderr, flush=True)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        print(f"\n\nATTENTION! before dec {x.shape} ", file=sys.stderr, flush=True)

        x = self.dec_emb(x)
        x = self.decoder(x[0])

        x = x.reshape(inputs.shape[0], num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        # print(f"\n\nATTENTION! masks: {masks}, mask shape: {masks.shape} ", file=sys.stderr, flush=True)
        iou_loss = 0
        if self.beta != 0:
            iou_loss = mask_iou(masks)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons, iou_loss, masks

    def step(self, batch, num_slots=None):
        if self.dataset == "celeba":
            imgs = batch[0]
        else:
            imgs = batch['image']
        result, _, iou_loss, _ = self(imgs, num_slots)
        loss = F.mse_loss(result, imgs)
        return loss, iou_loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, iou_loss = self.step(batch)
        self.log('Training MSE', loss)
        self.log('Training iou loss', iou_loss)

        loss = loss + iou_loss * self.beta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log('lr', sch.get_last_lr()[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou_loss = self.step(batch, num_slots=self.val_num_slots)
        self.log('Validation MSE', loss)
        self.log('Validation iou', iou_loss)

        if batch_idx == 0:
            imgs = batch['image']
            imgs = imgs[:8]

            result, recons, _, pred_masks = self(imgs, num_slots=self.val_num_slots)
            print("\n\nATTENTION! imgs: ", imgs.shape, file=sys.stderr, flush=True)
            print("\n\nATTENTION! recons: ", recons.shape, file=sys.stderr, flush=True)

            self.trainer.logger.experiment.log({
                'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
                'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)]
            })

            for i in range(self.val_num_slots):
                # print(f"\n\n\nATTENTION! {i} slot: ", recons[:, i], file=sys.stderr, flush=True)
                self.trainer.logger.experiment.log({
                    f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]

                })

        return loss

    def validation_epoch_end(self, outputdata):
        if self.current_epoch % 5 == 0:
            save_path = "./sa_autoencoder_end_to_end/" + f'{self.dataset}' + '/' + f'{self.task}'
            self.trainer.save_checkpoint(os.path.join(save_path,
                                                      f"{self.current_epoch}_{self.beta}_{self.task}_{self.dataset}_od_pretrained.ckpt"))

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=int(self.num_steps), pct_start=0.05)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        warmup_steps_pct = 0.02
        decay_steps_pct = 0.2
        scheduler_gamma = 0.5
        max_epochs = 100
        total_steps = max_epochs * len(self.train_dataloader)

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", }],
        )