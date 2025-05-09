# -*- coding: utf-8 -*-
import torch, torch.nn.functional as F
from gflownet_tlm.algos.base_trainer import BaseTrainer

class TLMTrainer(BaseTrainer):
    def compute_losses(self, states, actions, next_states, rewards, dones, ts):
        # ваша исходная реализация Trajectory Balance:
        # P_F(τ) = R/Z · P_B(τ) → loss_tb = (log P_F(τ) - log R + log Z - log P_B(τ))²
        # здесь Z можно опускать или считать константой
        # на практике берут batch-лосс по траекториям
        L_pi = ...  # скопируйте код из original trainer_tlm.py
        L_pb = ...  # обычно одно и то же что L_pi
        return {'loss_pi': L_pi, 'loss_pb': L_pb}

