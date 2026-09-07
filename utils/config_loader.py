"""
SentinelCold — Configuration Loader
Parses config.yaml and provides typed access to all settings.
"""

import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Typed accessor for the SentinelCold YAML configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

    # -- generic accessor -----------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot-notation, e.g. 'tier4.cv.n_splits'."""
        keys = dotted_key.split(".")
        val = self._raw
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    # -- convenience properties -----------------------------------------------
    @property
    def seed(self) -> int:
        return self._raw["seed"]

    @property
    def raw_data_path(self) -> Path:
        return Path(self._raw["paths"]["raw_data"])

    @property
    def output_dir(self) -> Path:
        return Path(self._raw["paths"]["output_dir"])

    @property
    def figures_dir(self) -> Path:
        return Path(self._raw["paths"]["figures_dir"])

    @property
    def tables_dir(self) -> Path:
        return Path(self._raw["paths"]["tables_dir"])

    @property
    def models_dir(self) -> Path:
        return Path(self._raw["paths"]["models_dir"])

    @property
    def reports_dir(self) -> Path:
        return Path(self._raw["paths"]["reports_dir"])

    @property
    def experiments_dir(self) -> Path:
        return Path(self._raw["paths"]["experiments_dir"])

    # -- tier-level accessors --------------------------------------------------
    @property
    def tier1(self) -> dict:
        return self._raw["tier1"]

    @property
    def tier2(self) -> dict:
        return self._raw["tier2"]

    @property
    def tier3(self) -> dict:
        return self._raw["tier3"]

    @property
    def tier4(self) -> dict:
        return self._raw["tier4"]

    @property
    def tier5(self) -> dict:
        return self._raw["tier5"]

    @property
    def tier6(self) -> dict:
        return self._raw["tier6"]

    @property
    def experiments(self) -> dict:
        return self._raw.get("experiments", {})

    # -- helpers ---------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create all output directories if they do not exist."""
        for d in [
            self.output_dir,
            self.figures_dir,
            self.tables_dir,
            self.models_dir,
            self.reports_dir,
            self.experiments_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict:
        """Return a deep-copy of the raw config for freezing in experiment logs."""
        import copy
        return copy.deepcopy(self._raw)

    def save_snapshot(self, dest: Path) -> None:
        """Write a frozen copy of the config to *dest*."""
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(self._raw, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config(path={self._path})"


# ---------------------------------------------------------------------------
# Module-level convenience — import once, use everywhere
# ---------------------------------------------------------------------------
_global_config: Optional[Config] = None


def load_config(path: str = "config.yaml") -> Config:
    """Load (or reload) the global configuration singleton."""
    global _global_config
    _global_config = Config(path)
    return _global_config


def get_config() -> Config:
    """Return the already-loaded global configuration."""
    if _global_config is None:
        raise RuntimeError("Config not loaded — call load_config() first.")
    return _global_config
