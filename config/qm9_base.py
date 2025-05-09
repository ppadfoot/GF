from ml_collections import ConfigDict

def get_base_config():
    cfg = ConfigDict()
    cfg.task        = 'qm9'
    # state_dim/action_dim задаются внутри QM9Env
    cfg.emb_dim     = 128
    cfg.seed        = 42
    cfg.batch_size  = 256
    cfg.max_steps   = 100_000
    cfg.gamma       = 0.99
    cfg.tau         = 0.01
    cfg.max_grad    = 5.0
    cfg.lr_pi       = 1e-4
    cfg.lr_pb       = 1e-4
    return cfg

