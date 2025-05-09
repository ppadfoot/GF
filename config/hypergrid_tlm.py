from .hypergrid_base import get_base_config

def get_config():
    cfg = get_base_config()
    cfg.method            = 'tlm'
    cfg.hidden_dim        = 221
    cfg.backward_approach = 'tlm'
    cfg.lr_tb             = 1e-3
    return cfg

