from .hypergrid_base import get_base_config

def get_config():
    cfg = get_base_config()
    cfg.method            = 'db'
    cfg.hidden_dim        = 221
    cfg.backward_approach = 'detailed_balance'
    cfg.lr_db             = 1e-3
    return cfg

