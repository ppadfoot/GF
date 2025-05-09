import random
import os

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric.data as gd
from torch_scatter import scatter

from mols.algo.config import Backward
from mols.envs.graph_building_env import GraphActionCategorical


def set_all_random(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


def shift_right(x: torch.Tensor, z=0):
    "Shift x right by 1, and put z in the first position"
    x = torch.roll(x, 1, dims=0)
    x[0] = z
    return x


def compute_trajs_prob(
    algo,
    model,
    batch: gd.Batch,
):
    """Compute the probabilities over the trajectories contained in the batch.
    Based on the `TrajectoryBalance.compute_batch_losses` function, and unnecessary lines are commented.

    Parameters
    ----------
    model: TrajectoryBalanceModel
        A GNN taking in a batch of graphs as input as per constructed by `construct_batch`.
        Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
    batch: gd.Batch
        batch of graphs inputs as per constructed by `construct_batch`
    num_bootstrap: int
        the number of trajectories for which the reward loss is computed. Ignored if 0."""
    dev = batch.x.device
    # A single trajectory is comprised of many graphs
    num_trajs = int(batch.traj_lens.shape[0])
    # log_rewards = batch.log_rewards
    # # Clip rewards
    # assert log_rewards.ndim == 1
    # clip_log_R = torch.maximum(log_rewards, torch.tensor(algo.global_cfg.algo.illegal_action_logreward, device=dev)).float()
    cond_info = getattr(batch, "cond_info", None)["encoding"]
    # invalid_mask = 1 - batch.is_valid

    # This index says which trajectory each graph belongs to, so
    # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
    # of length 4, trajectory 1 of length 3, and so on.
    batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
    # The position of the last graph of each trajectory
    traj_cumlen = torch.cumsum(batch.traj_lens, 0)
    final_graph_idx = traj_cumlen - 1
    # The position of the first graph of each trajectory
    first_graph_idx = shift_right(traj_cumlen)
    final_graph_idx_1 = torch.maximum(final_graph_idx - 1, first_graph_idx)

    fwd_cat: GraphActionCategorical  # The per-state cond_info
    batched_cond_info = cond_info[batch_idx] if cond_info is not None else None

    # Forward pass of the model, returns a GraphActionCategorical representing the forward
    # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
    with torch.no_grad():
        if algo.do_parameterize_p_b:
            fwd_cat, bck_cat, per_graph_out = model(batch, batched_cond_info)
        else:
            # if algo.model_is_autoregressive:
            #     fwd_cat, per_graph_out = model(batch, cond_info, batched=True)
            # else:
            #     fwd_cat, per_graph_out = model(batch, batched_cond_info)
            fwd_cat, per_graph_out = model(batch, batched_cond_info)
    # Retreive the reward predictions for the full graphs,
    # i.e. the final graph of each trajectory
    log_reward_preds = per_graph_out[final_graph_idx, 0]
    # if algo.cfg.do_predict_n:
    #     log_n_preds = per_graph_out[:, 1]
    #     log_n_preds[first_graph_idx] = 0
    # else:
    #     log_n_preds = None

    # Compute trajectory balance objective
    with torch.no_grad():
        log_Z = model.logZ(cond_info)[:, 0]
    # Compute the log prob of each action in the trajectory
    # if algo.cfg.do_correct_idempotent:
    #     # If we want to correct for idempotent actions, we need to sum probabilities
    #     # i.e. to compute P(s' | s) = sum_{a that lead to s'} P(a|s)
    #     # here we compute the indices of the graph that each action corresponds to, ip_lens
    #     # contains the number of idempotent actions for each transition, so we
    #     # repeat_interleave as with batch_idx
    #     ip_batch_idces = torch.arange(batch.ip_lens.shape[0], device=dev).repeat_interleave(batch.ip_lens)
    #     # Indicate that the `batch` corresponding to each action is the above
    #     ip_log_prob = fwd_cat.log_prob(batch.ip_actions, batch=ip_batch_idces)
    #     # take the logsumexp (because we want to sum probabilities, not log probabilities)
    #     # TODO: numerically stable version:
    #     p = scatter(ip_log_prob.exp(), ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum")
    #     # As a (reasonable) band-aid, ignore p < 1e-30, this will prevent underflows due to
    #     # scatter(small number) = 0 on CUDA
    #     log_p_F = p.clamp(1e-30).log()

    #     if algo.do_parameterize_p_b:
    #         # Now we repeat this but for the backward policy
    #         bck_ip_batch_idces = torch.arange(batch.bck_ip_lens.shape[0], device=dev).repeat_interleave(
    #             batch.bck_ip_lens
    #         )
    #         bck_ip_log_prob = bck_cat.log_prob(batch.bck_ip_actions, batch=bck_ip_batch_idces)
    #         bck_p = scatter(bck_ip_log_prob.exp(), bck_ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum")
    #         log_p_B = bck_p.clamp(1e-30).log()
    # else:

    # Else just naively take the logprob of the actions we took
    log_p_F = fwd_cat.log_prob(batch.actions)
    if algo.do_parameterize_p_b:
        log_p_B = bck_cat.log_prob(batch.bck_actions)

    if algo.do_parameterize_p_b:
        # If we're modeling P_B then trajectories are padded with a virtual terminal state sF,
        # zero-out the logP_F of those states
        log_p_F[final_graph_idx] = 0
        # if algo.cfg.variant == TBVariant.SubTB1 or algo.cfg.variant == TBVariant.DB:
        #     # Force the pad states' F(s) prediction to be R
        #     per_graph_out[final_graph_idx, 0] = clip_log_R

        # To get the correct P_B we need to shift all predictions by 1 state, and ignore the
        # first P_B prediction of every trajectory.
        # Our batch looks like this:
        # [(s1, a1), (s2, a2), ..., (st, at), (sF, None),   (s1, a1), ...]
        #                                                   ^ new trajectory begins
        # For the P_B of s1, we need the output of the model at s2.

        # We also have access to the is_sink attribute, which tells us when P_B must = 1, which
        # we'll use to ignore the last padding state(s) of each trajectory. This by the same
        # occasion masks out the first P_B of the "next" trajectory that we've shifted.
        log_p_B = torch.roll(log_p_B, -1, 0) * (1 - batch.is_sink)
        log_p_B[torch.isnan(log_p_B)] = 0
    else:
        log_p_B = batch.log_p_B
    assert log_p_F.shape == log_p_B.shape

    # if algo.cfg.n_loss == NLoss.TB:
    #     log_traj_n = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
    #     n_loss = algo._loss(log_traj_n + log_n_preds[final_graph_idx_1])
    # else:
    #     n_loss = algo.n_loss(log_p_B, log_n_preds, batch.traj_lens)

    # if algo.ctx.has_n() and algo.cfg.do_predict_n:
    #     analytical_maxent_backward = algo.analytical_maxent_backward(batch, first_graph_idx)
    #     if algo.do_parameterize_p_b:
    #         analytical_maxent_backward = torch.roll(analytical_maxent_backward, -1, 0) * (1 - batch.is_sink)
    # else:
    #     analytical_maxent_backward = None

    # if algo.cfg.backward_policy in [Backward.GSQL, Backward.GSQLA]:
    #     log_p_B = torch.zeros_like(log_p_B)
    #     nzf = torch.maximum(first_graph_idx, final_graph_idx - 1)
    #     if algo.cfg.backward_policy == Backward.GSQLA:
    #         log_p_B[nzf] = -batch.log_n
    #     else:
    #         log_p_B[nzf] = -log_n_preds[nzf]  # this is due to the fact that n(s_0)/n(s1) * n(s1)/ n(s2) = n(s_0)/n(s2) = 1 / n(s)
    #     # this is not final_graph_idx because we throw away the last thing
    # elif algo.cfg.backward_policy == Backward.MaxentA:
    #     log_p_B = analytical_maxent_backward

    if algo.do_parameterize_p_b:
        # Life is pain, log_p_B is one unit too short for all trajs

        log_p_B_unif = torch.zeros_like(log_p_B)
        for i, (s, e) in enumerate(zip(first_graph_idx, traj_cumlen)):
            log_p_B_unif[s : e - 1] = batch.log_p_B[s - i : e - 1 - i]

        if algo.cfg.backward_policy == Backward.Uniform:
            log_p_B = log_p_B_unif
    else:
        log_p_B_unif = log_p_B

    # if algo.cfg.backward_policy in [Backward.Maxent, Backward.GSQL]:
    #     log_p_B = log_p_B.detach()
    # This is the log probability of each trajectory
    traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
    traj_unif_log_p_B = scatter(log_p_B_unif, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
    traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

    # if algo.cfg.variant == TBVariant.SubTB1:
    #     # SubTB interprets the per_graph_out predictions to predict the state flow F(s)
    #     if algo.cfg.cum_subtb:
    #         traj_losses = algo.subtb_cum(log_p_F, log_p_B, per_graph_out[:, 0], clip_log_R, batch.traj_lens)
    #     else:
    #         traj_losses = algo.subtb_loss_fast(log_p_F, log_p_B, per_graph_out[:, 0], clip_log_R, batch.traj_lens)

    #     # The position of the first graph of each trajectory
    #     first_graph_idx = torch.zeros_like(batch.traj_lens)
    #     torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
    #     log_Z = per_graph_out[first_graph_idx, 0]
    # elif algo.cfg.variant == TBVariant.DB:
    #     F_sn = per_graph_out[:, 0]
    #     F_sm = per_graph_out[:, 0].roll(-1)
    #     F_sm[final_graph_idx] = clip_log_R
    #     transition_losses = algo._loss(F_sn + log_p_F - F_sm - log_p_B)
    #     traj_losses = scatter(transition_losses, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
    #     first_graph_idx = torch.zeros_like(batch.traj_lens)
    #     torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
    #     log_Z = per_graph_out[first_graph_idx, 0]
    # else:
    #     # Compute log numerator and denominator of the TB objective
    #     numerator = log_Z + traj_log_p_F
    #     denominator = clip_log_R + traj_log_p_B

    #     if algo.mask_invalid_rewards:
    #         # Instead of being rude to the model and giving a
    #         # logreward of -100 what if we say, whatever you think the
    #         # logprobablity of this trajetcory is it should be smaller
    #         # (thus the `numerator - 1`). Why 1? Intuition?
    #         denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

    #     if algo.cfg.epsilon is not None:
    #         # Numerical stability epsilon
    #         epsilon = torch.tensor([algo.cfg.epsilon], device=dev).float()
    #         numerator = torch.logaddexp(numerator, epsilon)
    #         denominator = torch.logaddexp(denominator, epsilon)
    #     traj_losses = algo._loss(numerator - denominator, algo.tb_loss)

    # # Normalize losses by trajectory length
    # if algo.length_normalize_losses:
    #     traj_losses = traj_losses / batch.traj_lens
    # if algo.reward_normalize_losses:
    #     # multiply each loss by how important it is, using R as the importance factor
    #     # factor = Rp.exp() / Rp.exp().sum()
    #     factor = -clip_log_R.min() + clip_log_R + 1
    #     factor = factor / factor.sum()
    #     assert factor.shape == traj_losses.shape
    #     # * num_trajs because we're doing a convex combination, and a .mean() later, which would
    #     # undercount (by 2N) the contribution of each loss
    #     traj_losses = factor * traj_losses * num_trajs

    # if algo.cfg.bootstrap_own_reward:
    #     num_bootstrap = num_bootstrap or len(log_rewards)
    #     reward_losses = algo._loss(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap], algo.reward_loss)

    #     reward_loss = reward_losses.mean() * algo.cfg.reward_loss_multiplier
    # else:
    #     reward_loss = 0

    # n_loss = n_loss.mean()
    # tb_loss = traj_losses.mean()
    # loss = tb_loss + reward_loss + algo.cfg.n_loss_multiplier * n_loss
    # bloss = -torch.sum(log_p_B[~log_p_B.isnan()])
    # info = {
    #     "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
    #     "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
    #     "reward_loss": reward_loss,
    #     "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
    #     "invalid_logprob": (invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4),
    #     "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
    #     "backward_vs_unif": (traj_unif_log_p_B - traj_log_p_B).pow(2).mean(),
    #     "logZ": log_Z.mean(),
    #     "loss": loss.item(),
    #     "n_loss": n_loss,
    #     "tb_loss": tb_loss.item(),
    #     "batch_entropy": -traj_log_p_F.mean(),
    #     "traj_lens": batch.traj_lens.float().mean(),
    # }
    # if algo.ctx.has_n() and algo.cfg.do_predict_n:
    #     info["n_loss_pred"] = scatter((log_n_preds - batch.log_ns) ** 2, batch_idx, dim=0, dim_size=num_trajs, reduce="sum").mean()
    #     info["n_final_loss"] = torch.mean((log_n_preds[final_graph_idx] - batch.log_n) ** 2)
    #     if algo.do_parameterize_p_b:
    #         info["n_loss_tgsql"] = torch.mean((-batch.log_n - traj_log_p_B) ** 2)
    #         d = analytical_maxent_backward - log_p_B
    #         d = d * d
    #         d[final_graph_idx] = 0
    #         info["n_loss_maxent"] = scatter(d, batch_idx, dim=0, dim_size=num_trajs, reduce="sum").mean()

    # return loss, bloss, info
    return traj_log_p_F, traj_log_p_B, traj_unif_log_p_B


def construct_batch(ctx, trajs, cond_info, log_rewards, do_parameterize_p_b):
    """Construct a batch from a list of trajectories and their information

    Parameters
    ----------
    trajs: List[List[tuple[Graph, GraphAction]]]
        A list of N trajectories.
    cond_info: Tensor
        The conditional info that is considered for each trajectory. Shape (N, n_info)
    log_rewards: Tensor
        The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
    Returns
    -------
    batch: gd.Batch
            A (CPU) Batch object with relevant attributes added
    """

    torch_graphs = [ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
    actions = [
        ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
    ]
    batch = ctx.collate(torch_graphs)
    batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
    # batch.results = [t["result"] for t in trajs]
    batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
    batch.actions = torch.tensor(actions)
    if do_parameterize_p_b:
        batch.bck_actions = torch.tensor(
            [
                ctx.GraphAction_to_ActionIndex(g, a)
                for g, a in zip(torch_graphs, [i for tj in trajs for i in tj["bck_a"]])
            ]
        )
        batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
    batch.log_rewards = log_rewards
    batch.cond_info = cond_info
    batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

    # compute_batch_losses expects these two optional values, if someone else doesn't fill them in, default to 0
    batch.num_offline = 0
    batch.num_online = 0
    return batch
