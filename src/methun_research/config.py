from dataclasses import dataclass


@dataclass
class EnhancedConfig:
    input_path: str = "data/cicids2017/cicids2017.csv"
    output_dir: str = "artifacts"
    epochs: int = 35
    batch_size: int = 512
    val_batch_size: int = 1024
    d_model: int = 160
    num_layers: int = 4
    heads: int = 10
    d_ff: int = 640
    dropout: float = 0.15
    lr: float = 0.002
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    focal_gamma: float = 1.8
    focal_alpha: float = 0.75
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    val_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    use_robust_scaling: bool = True
    undersampling_ratio: float = 0.12
    use_swa: bool = True
    swa_start: int = 20
    swa_freq: int = 3
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    gradient_accumulation_steps: int = 2
    use_multi_gpu: bool = False
    num_workers: int = 4


@dataclass
class CNNTransformerConfig:
    input_path: str = "data/cicids2017/cicids2017.csv"
    output_dir: str = "artifacts"
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 25
    batch_size: int = 256
    val_batch_size: int = 512
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    conv_channels: int = 96
    num_layers: int = 3
    num_heads: int = 8
    d_model: int = 192
    d_ff: int = 768
    dropout: float = 0.2
    undersampling_ratio: float = 0.15
    ig_steps: int = 32
    ig_samples: int = 512
    num_workers: int = 4

