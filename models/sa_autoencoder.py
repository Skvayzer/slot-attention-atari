import math
import os
import sys

# import tensorflow as tf
import pytorch_lightning as pl
import torch
import wandb

from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, ISAPosEmbeds, CoordQuantizer, MultiDspritesDecoder, TetrominoesDecoder, WaymoDecoder, WaymoEncoder
from modules.slot_attention import SlotAttentionBase, InvariantSlotAttention
from utils import spatial_broadcast, spatial_flatten, adjusted_rand_index, mask_iou
import torchvision

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
                 slot_size=64,
                 hidden_size=64,
                 dataset='',
                 task='',
                 invariance=True,
                 beta=0.01,
                 delta=5,
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
        self.delta = delta

        if dataset=='waymo':
            self.encoder = WaymoEncoder()
        else:
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
                *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
                  range(3)]
            )



        # Decoder
        if dataset in ['seaquest']:
            self.decoder_initial_size = (8, 8)
            self.decoder = Decoder(num_channels=slot_size)
        # elif dataset=='tetrominoes':
        #     self.decoder_initial_size = self.resolution
        #     self.decoder = TetrominoesDecoder(in_channels=self.resolution[0]*self.resolution[1],
        #                                         hidden_channels=256,
        #                                         out_channels=4)
        elif dataset=='tetrominoes':
            self.decoder_initial_size = self.resolution
            self.decoder = MultiDspritesDecoder(in_channels=self.slot_size,
                                   hidden_channels=self.hidden_size,
                                   out_channels=4,
                                   mode=dataset)
        elif dataset=='waymo':
            self.decoder_initial_size = (16, 24)
            self.decoder = WaymoDecoder(in_channels=self.slot_size,
                                        hidden_channels=self.hidden_size,
                                        out_channels=4,
                                        )
        if dataset=='waymo':
            self.enc_emb = ISAPosEmbeds(hidden_size, (16, 24))
        else:
            self.enc_emb = ISAPosEmbeds(hidden_size, self.resolution)

        self.dec_emb = ISAPosEmbeds(hidden_size, self.decoder_initial_size)
        self.h = nn.Linear(2, slot_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )

        if invariance:
            self.slot_attention = InvariantSlotAttention(num_slots=num_slots, iters=num_iters, dim=slot_size, delta=self.delta,
                                                hidden_dim=slot_size * 2, resolution=(16, 24), enc_hidden_size=hidden_size)
        else:
            self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                         hidden_dim=slot_size * 2)
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta
        # self.save_hyperparameters()

    def preprocess(self, encoded):
        x = spatial_flatten(encoded)
        x = self.enc_layer_norm(x)
        x = self.enc_mlp(x)
        x = self.norm_input(x)
        return x

    def forward(self, inputs, num_slots=None, test=False):
        x = self.encoder(inputs)
        # encoded torch.Size([32, 64, 128, 128])
        print(f"\n\nATTENTION! encoded {x.shape} ", file=sys.stderr, flush=True)

        # torch.autograd.set_detect_anomaly(True)
        # if not self.invariance:
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

        grid = self.dec_emb.grid.unsqueeze(dim=0).view(1, 1, -1, 2)

        if self.invariance:
            slots, S_p, S_r, S_s = self.slot_attention(x, n_s=num_slots)
            S_p = S_p.unsqueeze(dim=2)
            rel_grid = grid - S_p
            rel_grid /= self.delta
        else:
            slots = self.slot_attention(x, n_s=num_slots)
            rel_grid = grid
        print(f"\n\nATTENTION! before dec S_p: {S_p.shape} ", file=sys.stderr, flush=True)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        print(f"\n\nATTENTION! before dec pos emb: {grid.shape} ", file=sys.stderr, flush=True)
        # print(f"\n\nATTENTION! S_r: {S_r.shape} ", file=sys.stderr, flush=True)

        # print(f"\n\nATTENTION! rel_grid: {rel_grid.shape} ", file=sys.stderr, flush=True)

        # rel_grid_final = torch.zeros(rel_grid.shape).cuda()
        # rel_grid = torch.einsum('bskd,bsijd->bsijk', torch.inverse(S_r), grid - S_p)

        # S_r_inverse = torch.inverse(S_r)
        # print(f"\n\nATTENTION! S_r_inv: {S_r_inverse.shape} ", file=sys.stderr, flush=True)
        # rel_grid_final = torch.einsum("bsij,bskj->bski", S_r_inverse, rel_grid)
        rel_grid_final = rel_grid


        # x = x.reshape(*x.shape[:2], -1)
        # pos_emb = self.h(rel_grid_final).reshape(*x.shape[:2], -1)
        # temp = self.h(rel_grid_final)
        # print(f"\n\nATTENTION! temp: {temp.shape} ", file=sys.stderr, flush=True)
        # print(f"\n\nATTENTION! before dec: {x.shape} ", file=sys.stderr, flush=True)

        # pos_emb = self.h(rel_grid_final).reshape(1, x.shape[1], *self.decoder_initial_size)
        pos_emb = self.h(rel_grid_final).reshape(*x.shape[:2], *self.decoder_initial_size)

        print(f"\n\nATTENTION! pos_emb: {pos_emb.shape} ", file=sys.stderr, flush=True)
        print(f"\n\nATTENTION! before x: {x.shape} ", file=sys.stderr, flush=True)


        x = self.decoder(x + pos_emb)

        print(f"\n\nATTENTION! after dec: {x.shape} ", file=sys.stderr, flush=True)


        x = x.reshape(inputs.shape[0], num_slots, *x.shape[1:])
        print(f"\n\nATTENTION! reshaped: {x.shape} ", file=sys.stderr, flush=True)

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
        result, _, iou_loss, masks = self(imgs, num_slots)
        loss = F.mse_loss(result, imgs)
        return loss, iou_loss, masks

    def training_step(self, batch, batch_idx):
        print(f"\n\nATTENTION! Training step started ", file=sys.stderr, flush=True)

        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, iou_loss, _ = self.step(batch)
        self.log('Training MSE', loss)
        self.log('Training iou loss', iou_loss)


        loss = loss + iou_loss * self.beta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log('lr', sch.get_last_lr()[0], on_step=True, on_epoch=False)
        print(f"\n\nATTENTION! Training step ended ", file=sys.stderr, flush=True)

        return loss

    def validation_step(self, batch, batch_idx):
        print(f"\n\nATTENTION! Validation step started ", file=sys.stderr, flush=True)

        loss, iou_loss, pred_masks = self.step(batch, num_slots=self.val_num_slots)
        self.log('Validation MSE', loss)
        self.log('Validation iou', iou_loss)

        if self.dataset in ['tetrominoes']:
            true_masks = batch['mask']
            # print("\n\nATTENTION! true_masks: ", true_masks, true_masks.shape, file=sys.stderr, flush=True)
            # print("\n\nATTENTION! pred_masks: ", pred_masks, pred_masks.shape, file=sys.stderr, flush=True)

            pred_masks = pred_masks.view(*pred_masks.shape[:2], -1)
            true_masks = true_masks.view(*true_masks.shape[:2], -1)[:, 1:, :]
            # print("ATTENTION! MASKS (true/pred): ", true_masks.shape, pred_masks.shape, file=sys.stderr, flush=True)
            self.log('ARI', adjusted_rand_index(true_masks.float().cpu(), pred_masks.float().cpu()).mean())

        if batch_idx == 0:
            imgs = batch['image']
            imgs = imgs[:8]

            result, recons, _, pred_masks = self(imgs, num_slots=self.val_num_slots)
            # print("\n\nATTENTION! imgs: ", imgs.shape, file=sys.stderr, flush=True)
            # print("\n\nATTENTION! recons: ", recons.shape, file=sys.stderr, flush=True)

            self.trainer.logger.experiment.log({
                'images': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(imgs, -1, 1)],
                'reconstructions': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(result, -1, 1)]
            })

            for i in range(self.val_num_slots):
                # print(f"\n\n\nATTENTION! {i} slot: ", recons[:, i], file=sys.stderr, flush=True)
                self.trainer.logger.experiment.log({
                    f'{i} slot': [wandb.Image(x / 2 + 0.5) for x in torch.clamp(recons[:, i], -1, 1)]

                })

        print(f"\n\nATTENTION! Validation step ended ", file=sys.stderr, flush=True)

        # raise ValueError('A very specific bad thing happened.')

        return loss

    def validation_epoch_end(self, outputdata):
        if self.current_epoch % 10 == 0:
            save_path = "./sa_autoencoder_end_to_end/" + f'{self.dataset}' + '/' + f'{self.task}'
            # self.trainer.save_checkpoint(os.path.join(save_path, f"{self.current_epoch}_{self.beta}_{self.task}_{self.dataset}_od_pretrained.ckpt"))

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    #     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=int(self.num_steps), pct_start=0.05)
    #     return [optimizer], [scheduler]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        total_steps = 500_000
        steps_in_epoch = len(self.train_dataloader)
        print(f"\n\nATTENTION! steps_in_epoch: {steps_in_epoch} ", file=sys.stderr, flush=True)

        max_epochs = math.ceil(total_steps / steps_in_epoch)

        warmup_steps = 50_000
        warmup_epochs = warmup_steps / steps_in_epoch
        decay_steps = total_steps - warmup_steps

        decay_rate = 0.5

        print(f"\n\nATTENTION! other: {warmup_epochs} {decay_steps} ", file=sys.stderr, flush=True)

        # raise ValueError('A very specific bad thing happened.')


        def warm_and_decay_lr_scheduler(step: int):
            # warmup_steps = warmup_steps_pct * total_steps
            # decay_steps = decay_steps_pct * total_steps

            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            # factor *= decay_rate ** (step / decay_steps)

            # DEBUG
            # assert step < warmup_steps
            print(f"\n\nATTENTION! warm_and_decay_lr_scheduler step factor: {step} {factor} ", file=sys.stderr,
                  flush=True)

            return factor

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps + 1])
        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler2], milestones=[])

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
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
        #     *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
        #       range(3)]
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(2)],
            nn.Sequential(nn.Conv2d(hidden_size, slot_size, kernel_size=5, padding=(2, 2)), nn.ReLU())
        )
        self.decoder_initial_size = (8, 8)

        # Decoder
        if dataset == 'seaquest':
            self.decoder_initial_size = (8, 8)
            self.decoder = Decoder(num_channels=hidden_size)
        else:
            self.decoder_initial_size = self.resolution
            print(f"\n\nATTENTION! decoder_initial_size {self.decoder_initial_size} ", file=sys.stderr, flush=True)

            self.decoder = MultiDspritesDecoder(in_channels=self.slot_size,
                                                hidden_channels=self.hidden_size,
                                                out_channels=4,
                                                mode=dataset)

        self.enc_emb = PosEmbeds(slot_size, self.resolution)
        self.dec_emb = PosEmbeds(slot_size, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )


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
        # print(f"\n\nATTENTION! batch 1 image  {imgs.shape} {imgs[0]} ", file=sys.stderr, flush=True)

        result, _, iou_loss, pred_masks = self(imgs, num_slots)
        # print(f"\n\nATTENTION! result image {result.shape} {result[0]} ", file=sys.stderr, flush=True)

        loss = F.mse_loss(result, imgs)
        # print(f"\n\nATTENTION! loss {loss} ", file=sys.stderr, flush=True)

        return loss, iou_loss, pred_masks

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, iou_loss, _ = self.step(batch)
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
        loss, iou_loss, pred_masks = self.step(batch, num_slots=self.val_num_slots)
        self.log('Validation MSE', loss)
        self.log('Validation iou', iou_loss)

        true_masks = batch['mask']
        # print("\n\nATTENTION! true_masks: ", true_masks, true_masks.shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! pred_masks: ", pred_masks, pred_masks.shape, file=sys.stderr, flush=True)

        pred_masks = pred_masks.view(*pred_masks.shape[:2], -1)
        true_masks = true_masks.view(*true_masks.shape[:2], -1)[:, 1:, :]
        # print("ATTENTION! MASKS (true/pred): ", true_masks.shape, pred_masks.shape, file=sys.stderr, flush=True)
        self.log('ARI', adjusted_rand_index(true_masks.float().cpu(), pred_masks.float().cpu()).mean())

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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
    #
    #     warmup_steps_pct = 0.02
    #     decay_steps_pct = 0.2
    #     scheduler_gamma = 0.5
    #     max_epochs = 100
    #     total_steps = max_epochs * len(self.train_dataloader)
    #
    #     warmup_steps = 10_000
    #     decay_steps = 100_000
    #     decay_rate = 0.5
    #     total_steps = 500_000
    #
    #     def warm_and_decay_lr_scheduler(step: int):
    #         # warmup_steps = warmup_steps_pct * total_steps
    #         # decay_steps = decay_steps_pct * total_steps
    #         assert step < total_steps
    #         if step < warmup_steps:
    #             factor = step / warmup_steps
    #         else:
    #             factor = 1
    #         factor *= scheduler_gamma ** (step / decay_steps)
    #         return factor
    #
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
    #
    #     return (
    #         [optimizer],
    #         [{"scheduler": scheduler, "interval": "step", }],
    #     )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        total_steps = 50_000
        steps_in_epoch = len(self.train_dataloader)
        print(f"\n\nATTENTION! steps_in_epoch: {steps_in_epoch} ", file=sys.stderr, flush=True)

        max_epochs = math.ceil(total_steps / steps_in_epoch)

        warmup_steps = 5_000
        warmup_epochs = warmup_steps / steps_in_epoch
        decay_steps = total_steps - warmup_steps

        decay_rate = 0.5

        print(f"\n\nATTENTION! other: {warmup_epochs} {decay_steps} ", file=sys.stderr, flush=True)

        # raise ValueError('A very specific bad thing happened.')


        def warm_and_decay_lr_scheduler(step: int):
            # warmup_steps = warmup_steps_pct * total_steps
            # decay_steps = decay_steps_pct * total_steps

            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            # factor *= decay_rate ** (step / decay_steps)

            # DEBUG
            # assert step < warmup_steps
            print(f"\n\nATTENTION! warm_and_decay_lr_scheduler step factor: {step} {factor} ", file=sys.stderr,
                  flush=True)

            return factor

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps + 1])
        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler2], milestones=[])

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )