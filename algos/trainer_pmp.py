# -*- coding: utf-8 -*-
import torch, torch.nn as nn
from gflownet_tlm.algos.base_trainer   import BaseTrainer
from gflownet_tlm.models.embedding      import EmbeddingNet
from gflownet_tlm.models.lambda_net     import LambdaNet
from gflownet_tlm.models.surrogate_reward import SurrogateReward

class PMPTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.phi     = EmbeddingNet(cfg.state_dim, cfg.emb_dim).to(self.device)
        self.l_net   = LambdaNet(cfg.emb_dim,   cfg.hidden_dim).to(self.device)
        self.R_hat   = SurrogateReward(cfg.emb_dim).to(self.device)

        self.opt_phi    = torch.optim.Adam(self.phi.parameters(),    lr=cfg.lr_phi)
        self.opt_lambda = torch.optim.Adam(self.l_net.parameters(),  lr=cfg.lr_lambda)
        self.opt_Rhat   = torch.optim.Adam(self.R_hat.parameters(),  lr=cfg.lr_Rhat)

    def compute_losses(self, states, actions, next_states, rewards, dones, ts):
        emb_s  = self.phi(states)
        emb_ns = self.phi(next_states)

        # boundary ∇ log R
        with torch.no_grad():
            Rhat_ns = self.R_hat(emb_ns)
            grad_R  = torch.autograd.grad(Rhat_ns.sum(), emb_ns)[0]

        lam_next = self.l_net(emb_ns, ts+1).detach()
        r_ = rewards.unsqueeze(-1); d_ = dones.unsqueeze(-1)
        td_target = torch.where(d_, grad_R, r_ + self.cfg.gamma * lam_next)

        loss_lambda = nn.functional.mse_loss(self.l_net(emb_s, ts), td_target)
        loss_Rhat   = nn.functional.mse_loss(self.R_hat(emb_ns), r_)

        logp    = self.policy.log_prob(states, actions)
        delta   = (emb_ns - emb_s)
        ham     = logp - (lam_next * delta).sum(dim=1)
        loss_pi = -ham.mean()

        loss_pb = -self.pb.log_prob(next_states, actions).mean()

        return {
            'loss_lambda': loss_lambda,
            'loss_Rhat':   loss_Rhat,
            'loss_pi':     loss_pi,
            'loss_pb':     loss_pb
        }

    def update(self, batch):
        states, actions, next_states, rewards, dones, ts = [b.to(self.device) for b in batch]
        losses = self.compute_losses(states, actions, next_states, rewards, dones, ts)

        # 1) λ
        self.opt_lambda.zero_grad()
        losses['loss_lambda'].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.l_net.parameters(), self.cfg.max_grad)
        self.opt_lambda.step()

        # 2) R̂
        self.opt_Rhat.zero_grad()
        losses['loss_Rhat'].backward(retain_graph=True)
        self.opt_Rhat.step()

        # 3) φ
        self.opt_phi.zero_grad()
        (losses['loss_lambda'] + losses['loss_Rhat']).backward(retain_graph=True)
        self.opt_phi.step()

        # 4) π_F и π_B
        metrics = super().update(batch)
        metrics.update({
            'loss_lambda': losses['loss_lambda'].item(),
            'loss_Rhat':   losses['loss_Rhat'].item()
        })
        return metrics

