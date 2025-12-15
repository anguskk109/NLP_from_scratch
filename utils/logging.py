# utils/logging.py
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


class JSONLogger:
    def __init__(self, log_dir: str, project_name: str = "experiment"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{project_name}_{timestamp}"
        self.log_file = os.path.join(self.log_dir, f"{self.run_name}_metrics.jsonl")
        self.config_file = os.path.join(self.log_dir, f"{self.run_name}_config.yaml")
        
        print(f"Logging to: {self.log_file}")

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append metrics as a JSON line."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save config as YAML (requires PyYAML) or JSON."""
        try:
            import yaml
            with open(self.config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback to JSON
            config_json = os.path.splitext(self.config_file)[0] + ".json"
            with open(config_json, "w") as f:
                json.dump(config, f, indent=2)

    def get_run_dir(self) -> str:
        return os.path.dirname(self.log_file)

_logger = None


def setup_json_logging(
    log_dir: str = "logs",
    project_name: str = "nlp-transformer-practice",
    config: Optional[Dict[str, Any]] = None,
) -> JSONLogger:
    """
    Initialize JSON logger.
    Returns logger instance for direct use.
    """
    global _logger
    _logger = JSONLogger(log_dir=log_dir, project_name=project_name)
    if config is not None:
        _logger.save_config(config)
    return _logger


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log metrics using global logger."""
    global _logger
    if _logger is not None:
        _logger.log(metrics)
    else:
        # Fallback to stdout
        print(f"Metrics: {metrics}")