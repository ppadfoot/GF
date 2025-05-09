# -*- coding: utf-8 -*-
import random, numpy as np, torch, torch.nn as nn
from gflownet_tlm.models.mlp_policy import MLPPolicy

class BaseTrainer:
    """
    Общий тренер: forward/backward-policy + оптимизаторы.
    Наследники только реализуют compute_losses().
    """
    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

        # π_F
        self.policy = MLPPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        # π_B
        self.pb     = MLPPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)

        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_pi)
        lr_pb = getattr(cfg, 'lr_pb', None) or getattr(cfg, 'lr_tb', None) or getattr(cfg, 'lr_db', None)
        self.opt_pb = torch.optim.Adam(self.pb.parameters(),     lr=lr_pb)

    def compute_losses(self, states, actions, next_states, rewards, dones, ts):
        raise NotImplementedError

    def update(self, batch):
        states, actions, next_states, rewards, dones, ts = [b.to(self.device) for b in batch]
        losses = self.compute_losses(states, actions, next_states, rewards, dones, ts)

        # forward update
        self.opt_pi.zero_grad(); losses['loss_pi'].backward()
        if self.cfg.max_grad: nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad)
        self.opt_pi.step()

        # backward update
        self.opt_pb.zero_grad(); losses['loss_pb'].backward()
        if self.cfg.max_grad: nn.utils.clip_grad_norm_(self.pb.parameters(), self.cfg.max_grad)
        self.opt_pb.step()

        return {k:v.item() for k,v in losses.items()}

