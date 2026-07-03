class FLAGS:
    learning_rate: float = 0.0001
    weight_decay: float = 5e-4
    dimension: int = 8
    hidden_dim: int = 32
    num_heads: int = 1
    dropout: float = 0.7
    edge_dropout: float = 0.5
    noise_std: float = 0.05
    l2_latent: float = 0.01
    epsilon: float = 0.001
    epochs: int = 500
    patience: int = 40
    log_interval: int = 10
    prop_test: float = 10.0
    prop_val: float = 5.0
