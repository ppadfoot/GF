from .hypergrid_base import get_base_config

def get_config():
    cfg = get_base_config()
    cfg.method     = 'pmp'
    cfg.emb_dim    = 128
    cfg.hidden_dim = 128
    cfg.lr_lambda  = 1e-3
    cfg.lr_phi     = 1e-3
    cfg.lr_Rhat    = 1e-3
    return cfg

