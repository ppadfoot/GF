from ml_collections import ConfigDict

def get_base_config():
    cfg = ConfigDict()
    cfg.task        = 'hypergrid'
    cfg.state_dim   = 4
    cfg.action_dim  = 4
    cfg.seed        = 42
    cfg.batch_size  = 256
    cfg.max_steps   = 100_000
    cfg.gamma       = 0.99
    cfg.tau         = 0.01
    cfg.max_grad    = 5.0
    cfg.lr_pi       = 1e-4
    cfg.lr_pb       = 1e-4
    return cfg

