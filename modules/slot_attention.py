import math
import sys

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from utils import spatial_flatten, build_grid
from modules import PosEmbeds



def norm_prob(mus, logsigmas, values):
    mus = torch.unsqueeze(mus, 2)
    logsigmas = torch.unsqueeze(logsigmas, 2)
    values = torch.unsqueeze(values, 1)
    var = torch.exp(logsigmas)**2
    log_prob =  (-((values - mus) ** 2) / (2 * var)).sum(dim=-1) - logsigmas.sum(dim=-1) - values.shape[-1] * math.log(math.sqrt((2 * math.pi)))
    return torch.exp(log_prob)


class InvariantSlotAttention(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128, resolution=(128, 128), enc_hidden_size=64):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.resolution = resolution
        self.dim = dim


        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        # self.slots_mu = nn.Sequential(
        #     nn.Linear(dim, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 2),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.resolution[0]*self.resolution[1] * 2, dim)
        # )
        #
        #
        # self.slots_logsigma = nn.Sequential(
        #     nn.Linear(dim, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 2),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.resolution[0]*self.resolution[1] * 2, dim)
        # )


        self.enc_emb = PosEmbeds(enc_hidden_size, self.resolution, mode='isa')
        self.abs_grid = self.enc_emb.grid
        self.abs_grid_flattened = self.abs_grid.reshape(self.abs_grid.shape[1] * self.abs_grid.shape[2], self.abs_grid.shape[-1]).cuda()

        self.enc_layer_norm = nn.LayerNorm(enc_hidden_size)
        self.enc_mlp = nn.Sequential(
            nn.Linear(enc_hidden_size, enc_hidden_size),
            nn.ReLU(),
            nn.Linear(enc_hidden_size, dim)
        )



        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.g = nn.Linear(2, enc_hidden_size)
        self.f = nn.Sequential(
            nn.LayerNorm(enc_hidden_size),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def encode_pos(self, encoded, grid):
        x = self.enc_emb(encoded, grid)
        x = spatial_flatten(x[0])
        x = self.enc_layer_norm(x)
        x = self.enc_mlp(x)
        x = self.norm_input(x)

        return x

    def forward(self, inputs, n_s=None, grid=None,  *args, **kwargs):
        encoded_pos = self.encode_pos(inputs, self.abs_grid)

        b, n, d, device = *encoded_pos.shape, encoded_pos.device
        if n_s is None:
            n_s = self.num_slots
        print(f"\n\nATTENTION! ns: {n_s} ", file=sys.stderr, flush=True)


        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device = device)

        # slots_mu = self.slots_mu(inputs)
        # print(f"\n\nATTENTION! slots_mu shape: {slots_mu.shape} ", file=sys.stderr, flush=True)
        # slots_logsigma = self.slots_logsigma(inputs)
        # slots_mu, slots_log_sigma = slots_mu.sum(axis=0), slots_logsigma.sum(axis=0)
        # slots_mu, slots_log_sigma = slots_mu.reshape((1, 1, self.dim)), slots_log_sigma.reshape(
        #     (1, 1, self.dim))
        # # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        # slots_init = torch.randn((b, n_s, self.dim), device = device)
        # slots_init = slots_init.type_as(inputs)
        # slots = slots_mu + slots_log_sigma * slots_init

        # inputs = self.norm_input(inputs)
        S_p = 2 * torch.rand((self.num_slots, 2)) - 1
        for t in range(1, self.iters + 1):
            slots_prev = slots

            slots = self.norm_slots(slots)

            # Computes relative grids per slot, and associated key, value embeddings
            rel_grid = (self.abs_grid )#- S_p)
            encoded_pos = self.encode_pos(inputs, rel_grid.cuda())
            k, v = self.to_k(encoded_pos), self.to_v(encoded_pos)
            print(f"\n\nATTENTION! k v: {k.shape} {v.shape} ", file=sys.stderr, flush=True)
            print(f"\n\nATTENTION! bas grid: {self.abs_grid.shape}", file=sys.stderr, flush=True)

            # k = self.f(self.to_k(inputs) + self.g(rel_grid))
            # v = self.f(self.to_v(inputs) + self.g(rel_grid))

            # Inverted dot production attention.
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            updates = torch.einsum('bjd,bij->bid', v, attn)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            print(f"\n\nATTENTION! attn: {attn.shape} ", file=sys.stderr, flush=True)
            # abs_grid_expanded = self.abs_grid_flattened.expand(b, self.abs_grid_flattened.shape[0], self.abs_grid_flattened.shape[1])
            print(f"\n\nATTENTION! a: {attn[:, 0].unsqueeze(dim=1).shape} ", file=sys.stderr, flush=True)

            print(f"\n\nATTENTION! sp fl grid: {self.abs_grid_flattened.shape} ", file=sys.stderr, flush=True)

            # Updates Sp, Ss and slots.
            for i in range(n_s):
                S_p[i] = (attn[:, i] @ self.abs_grid_flattened).sum(dim=-1, keepdim=True) / attn.sum(dim=-1, keepdim=True)
            print(f"\n\nATTENTION! S_p: {S_p.shape} ", file=sys.stderr, flush=True)

            # S_s = (((attn + self.eps)*(grid - S_p)**2).sum(dim=-1, keepdim=True)/(attn + self.eps).sum(dim=-1, keepdim=True))**0.5
            # v1, v2 = WPCA().fit_transform(self.abs_grid, attn)
            # S_r = postprocess(v1, v2)

            if t < self.iters + 1:
                slots = self.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d)
                )

                slots = slots.reshape(b, -1, d)
                slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionBase(nn.Module):
    """
    Slot Attention module
    """

    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def step(self, slots, k, v, b, n, d, device, n_s):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

    def forward(self, inputs, n_s=None, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        if n_s is None:
            n_s = self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots = self.step(slots, k, v, b, n, d, device, n_s)
        slots = self.step(slots.detach(), k, v, b, n, d, device, n_s)

        return slots


class SlotAttentionGMM(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim*2))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim*2))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_q_sig = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)
        self.gru_logsigma = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim

        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.mlp_sigma = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim*2)
        self.norm_mu = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        
        self.mlp_out = nn.Sequential(
            nn.Linear(dim*2, hidden_dim*2),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim*2, dim)
        )
        
    def step(self, slots, k, v, b, n, d, device, n_s, pi_cl):
        slots_prev = slots

        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)
        
                
        # probs = norm_prob(slots_mu, slots_logsigma, k) * pi_cl
        # if torch.isnan(probs).any():
        #     print('PROBS Nan appeared')
        # probs = probs / (probs.sum(dim=1, keepdim=True) + self.eps)
        # if torch.isnan(probs).any():
        #     print('PROBS2 Nan appeared')
        # probs = probs / (probs.sum(dim=-1, keepdim=True) + self.eps)
        # if torch.isnan(probs).any():
        #     print('PROBS3 Nan appeared')
        
        #dots = torch.einsum('bid,bjd->bij', q_mu / torch.exp(q_logsigma)**2, k) * self.scale
        dots = ((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) ** 2 / torch.unsqueeze(torch.exp(q_logsigma)**2, 2)).sum(dim=-1) * self.scale
        dots_exp = (torch.exp(-dots) + self.eps) * pi_cl
        attn = dots_exp / dots_exp.sum(dim=1, keepdim=True)
        #attn = (dots.softmax(dim=1) + self.eps)*pi_cl
        #attn = attn / attn.sum(dim=1, keepdim=True)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = updates_mu.reshape(b, -1, d)
        updates_mu = updates_mu + self.mlp_mu(self.norm_mu(updates_mu))
        if torch.isnan(updates_mu).any():
            print('updates_mu Nan appeared')
        
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', ((torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2))**2 + self.eps, attn)))
        #updates_logsigma = updates_logsigma + self.mlp_sigma(self.norm_sigma(updates_logsigma))
        if torch.isnan(updates_logsigma).any():
            print('updates_logsigma Nan appeared')
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)
        
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = pi_cl_new / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)

        # updates = torch.einsum('bjd,bij->bid', v, attn)

        # slots = self.gru(
        #     updates.reshape(-1, d*2),
        #     slots_prev.reshape(-1, d*2)
        # )
        if torch.isnan(slots).any():
            print('gru Nan appeared')
        
        if torch.isnan(slots).any():
            print('MLP Nan appeared')
        return slots, pi_cl_new

    def forward(self, inputs, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.num_slots
        
        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots, pi_cl = self.step(slots, k, v, b, n, d, device, n_s, pi_cl)
        slots, pi_cl = self.step(slots.detach(), k, v, b, n, d, device, n_s, pi_cl)

        return self.mlp_out(slots)

    
class SlotAttention(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.koefs = nn.Parameter(torch.ones(iters, 2))
        
    def step(self, slots, k, v, b, n, d, device, n_s):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

    def forward(self, inputs, pos, mlp, norm, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        for i in range(self.iters):
            koefs = F.softmax(self.koefs, dim=-1)
            cur_inputs = inputs * self.koefs[i, 0] + pos * self.koefs[i, 1]
            cur_inputs = mlp(norm(cur_inputs))
            
            cur_inputs = self.norm_input(cur_inputs)      
            k, v = self.to_k(cur_inputs), self.to_v(cur_inputs)
            
            slots = self.step(slots, k, v, b, n, d, device, n_s)
        slots = self.step(slots.detach(), k, v, b, n, d, device, n_s)

        return slots

if __name__ == "__main__":
    slotattention = InvariantSlotAttention(num_slots=10, dim=64)
    state_dict = torch.load("/home/alexandr_ko/quantised_sa_od/clevr10_sp")
    key:str
    state_dict = {key[len('slot_attention.'):]:state_dict[key] for key in state_dict if key.startswith('slot_attention')}
    slotattention.load_state_dict(state_dict=state_dict)
    print("DOne")