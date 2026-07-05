"""Load and validate the YAML configuration file."""
from pathlib import Path
import yaml


def load_config(path: str | Path = "experiments/configs/base_config.yaml") -> dict:
    """
    Read the project config from a YAML file.

    Parameters
    ----------
    path : absolute or relative path to the YAML file

    Returns
    -------
    Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg
