# -*- coding: utf-8 -*-
"""
BaseTrainer — универсальный класс для обучения GFlowNet-методов:
Flow Matching, Detailed Balance, Trajectory Balance, TLM и PMP.

Он создаёт:
  1) Две политики (forward и backward) на основе MLPPolicy.
  2) Оптимизаторы для обеих политик.
  3) Метод update(), который вызывают наследники.

Наследники должны переопределять только compute_losses().
"""

import random
import numpy as np
import torch
import torch.nn as nn

# Ваш класс MLPPolicy: принимает state_dim, action_dim, hidden_dim
from gflownet_tlm.models.mlp_policy import MLPPolicy

class BaseTrainer:
    def __init__(self, cfg):
        """
        Инициализация:
          cfg: атрибуты config-объекта, например:
            - cfg.state_dim (int)
            - cfg.action_dim (int)
            - cfg.hidden_dim (int)
            - cfg.lr_pi, cfg.lr_pb
            - cfg.seed, cfg.max_grad
        """
        self.cfg = cfg

        # устройство — CUDA если доступен, иначе CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # фиксируем сида для reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # 1) forward-policy: π(a|s)
        self.policy = MLPPolicy(
            state_dim  = cfg.state_dim,
            action_dim = cfg.action_dim,
            hidden_dim = cfg.hidden_dim
        ).to(self.device)

        # 2) backward-policy: P_B(a|s')
        self.pb = MLPPolicy(
            state_dim  = cfg.state_dim,
            action_dim = cfg.action_dim,
            hidden_dim = cfg.hidden_dim
        ).to(self.device)

        # 3) оптимизатор для forward-policy
        self.opt_pi = torch.optim.Adam(
            params = self.policy.parameters(),
            lr     = cfg.lr_pi
        )

        # 4) оптимизатор для backward-policy
        #    lr_pb может быть одно из cfg.lr_pb, cfg.lr_tb или cfg.lr_db
        lr_pb = getattr(cfg, 'lr_pb', None) or getattr(cfg, 'lr_tb', None) or getattr(cfg, 'lr_db', None)
        self.opt_pb = torch.optim.Adam(
            params = self.pb.parameters(),
            lr     = lr_pb
        )

    def compute_losses(self, states, actions, next_states, rewards, dones, ts):
        """
        Заглушка: наследники переопределяют этот метод.

        Входные тензоры:
          - states:      [B, state_dim]  — batch текущих состояний
          - actions:     [B] или [B, ?]    — batch действий
          - next_states: [B, state_dim]  — batch следующих состояний
          - rewards:     [B]             — batch вознаграждений
          - dones:       [B] (bool)      — маркеры терминала
          - ts:          [B] (int)       — номера временного шага

        Должен вернуть словарь:
          {
            'loss_pi': Tensor,  # loss для forward-policy
            'loss_pb': Tensor,  # loss для backward-policy
            # ... и любые другие лоссы, которые хотите логировать
          }
        """
        raise NotImplementedError("compute_losses() must be implemented in subclass.")

    def update(self, batch):
        """
        Универсальный update:
          1) Распаковать batch (batch = tuple of tensors).
          2) Вызвать compute_losses() у наследника.
          3) Сделать backward+step для forward-policy и backward-policy.
          4) Вернуть скалярные метрики для логгирования.
        """
        # 1) распаковка и перенос на device
        states, actions, next_states, rewards, dones, ts = [
            b.to(self.device) for b in batch
        ]

        # 2) конкретные потери от наследника
        losses = self.compute_losses(states, actions, next_states, rewards, dones, ts)
        # losses — dict {'loss_pi': Tensor, 'loss_pb': Tensor, ...}

        # 3) update forward-policy
        self.opt_pi.zero_grad()
        losses['loss_pi'].backward()
        if self.cfg.max_grad is not None:
            # ограничение нормы градиента
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad)
        self.opt_pi.step()

        # 4) update backward-policy
        self.opt_pb.zero_grad()
        losses['loss_pb'].backward()
        if self.cfg.max_grad is not None:
            nn.utils.clip_grad_norm_(self.pb.parameters(), self.cfg.max_grad)
        self.opt_pb.step()

        # 5) вернуть все числовые лоссы для логгирования
        return {k: v.item() for k, v in losses.items()}

