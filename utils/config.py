# utils/config.py
import yaml
from typing import Any, Dict, Type
import os

class BaseConfig:
    def __init__(self, **kwargs):
        # Shared transformer hyperparameters
        self.hidden_size = int(kwargs.get("hidden_size", 256))
        self.num_layers = int(kwargs.get("num_layers", 4))
        self.num_heads = int(kwargs.get("num_heads", 4))
        self.ffn_hidden_size = int(kwargs.get("ffn_hidden_size", 512))
        self.dropout = float(kwargs.get("dropout", 0.1))
        self.max_seq_len = int(kwargs.get("max_seq_len", 128))
        self.vocab_size = int(kwargs.get("vocab_size", 50257))
        self.pad_token_id = int(kwargs.get("pad_token_id", 0))
        self.initializer_range = float(kwargs.get("initializer_range", 0.02))
        self.layer_norm_eps = float(kwargs.get("layer_norm_eps", 1e-5))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for logging"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def save_to_yaml(self, filepath: str) -> None:
        """Save config to a YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseConfig":
        """Load config from a YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class BertConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlm_probability: float = kwargs.get("mlm_probability", 0.15)


class GPTConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # GPT-like does not need any beyond base for now
        pass


class T5Config(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_corruption_rate: float = kwargs.get("span_corruption_rate", 0.15)

CONFIG_CLASSES = {
    "bert": BertConfig,
    "gpt": GPTConfig,
    "t5": T5Config,
}

def get_config_class(model_type: str) -> Type[BaseConfig]:
    if model_type not in CONFIG_CLASSES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(CONFIG_CLASSES.keys())}")
    return CONFIG_CLASSES[model_type]