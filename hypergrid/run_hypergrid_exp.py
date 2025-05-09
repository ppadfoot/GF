#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example:

>> python run_hypergrid_exp.py \
     --general experiments/config/general.py:3 \
     --env     experiments/config/hypergrid.py:standard \
     --algo    experiments/config/algo.py:soft_ql \
     --algo.munchausen.alpha 0.00
"""

from absl import app
from absl import flags
from ml_collections import config_flags
import random
import os
import numpy as np
import torch

from gfn.gym import HyperGrid

from experiments.train_softdqn   import train_softdqn
from experiments.train_uniform   import train_uniform
from experiments.train_perfect   import train_perfect
from experiments.train_baseline  import train_baseline

FLAGS = flags.FLAGS

# Подключаем файлы конфигурации
config_flags.DEFINE_config_file("general")
config_flags.DEFINE_config_file("env")
config_flags.DEFINE_config_file("algo")

# Сопоставление названий алгоритмов и функций обучения
train_fns = {
    "SoftDQN":            train_softdqn,
    "TrajectoryBalance":  train_baseline,
    "DetailedBalance":    train_baseline,
    "SubTrajectoryBalance": train_baseline,
    "Uniform":            train_uniform,
    "GroundTruth":        train_perfect,
}

def set_seed(seed: int, is_cuda: bool = False):
    """Фиксируем все сиды для повторяемости."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(_):
    # Загружаем параметры из флагов
    general_args = FLAGS.general
    env_args     = FLAGS.env
    algo_args    = FLAGS.algo

    # Проверяем seed
    if general_args.seed == 0:
        raise ValueError("Seed should be >0!")
    seed = general_args.seed
    is_cuda = (general_args.device == "cuda")
    set_seed(seed, is_cuda=is_cuda)

    # (Опционально) запускаем wandb
    use_wandb = getattr(general_args, "wandb_project", "") != ""
    if use_wandb:
        import wandb
        wandb.init(project=general_args.wandb_project)
        wandb.config.update(general_args)
        wandb.config.update(env_args)
        wandb.config.update(algo_args)

    # Инициализируем среду HyperGrid
    env = HyperGrid(
        env_args.ndim,
        env_args.height,
        env_args.R0,
        env_args.R1,
        env_args.R2,
        device_str=general_args.device,
    )

    # Директория для результатов
    env_name = f"{env_args.reward_type}_{env_args.ndim}_{env_args.height}"
    os.makedirs(f"grid_results_{env_name}", exist_ok=True)

    # Формируем имя эксперимента
    experiment_name = f"grid_results_{env_name}/{seed}_{algo_args.name}"

    # Включаем детектирование NaN
    torch.autograd.set_detect_anomaly(True, check_nan=True)

    # Запускаем нужную функцию обучения
    train_fns[algo_args.name](env, experiment_name, general_args, algo_args)

if __name__ == "__main__":
    app.run(main)

