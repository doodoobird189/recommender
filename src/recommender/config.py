"""Central configuration: filesystem layout and model hyperparameters.

Paths are resolved from environment variables (so Docker / CI can override them) and
default to locations relative to the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Paths:
    raw: Path = field(default_factory=lambda: _env_path("RECSYS_RAW_DIR", "data/ml-latest"))
    processed: Path = field(
        default_factory=lambda: _env_path("RECSYS_PROCESSED_DIR", "data/processed")
    )
    artifacts: Path = field(default_factory=lambda: _env_path("RECSYS_ARTIFACTS_DIR", "artifacts"))
    reports: Path = field(default_factory=lambda: _env_path("RECSYS_REPORTS_DIR", "reports"))

    # processed (intermediate, not needed at serving time)
    @property
    def interactions(self) -> Path:
        return self.processed / "interactions.parquet"

    # serving artifacts
    @property
    def items(self) -> Path:
        return self.artifacts / "items.parquet"

    @property
    def genome(self) -> Path:
        return self.artifacts / "genome.npz"

    @property
    def item_factors(self) -> Path:
        return self.artifacts / "als_item_factors.npy"

    @property
    def model_card(self) -> Path:
        return self.artifacts / "model.json"

    @property
    def sample_users(self) -> Path:
        return self.artifacts / "sample_users.parquet"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics.json"


@dataclass(frozen=True)
class DataConfig:
    min_user_interactions: int = 20
    min_item_interactions: int = 20
    n_sample_users: int = 500  # MovieLens users shipped with the app for the "random user" demo


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 64
    regularization: float = 0.1
    iterations: int = 20
    # Hu, Koren & Volinsky (2008): confidence c_ui = 1 + alpha * r_ui
    alpha: float = 1.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalConfig:
    test_fraction: float = 0.1
    n_eval_users: int = 5000
    k_values: tuple[int, ...] = (5, 10, 20)
    # items with fewer than this many *training* interactions count as "cold"
    cold_item_threshold: int = 5
    recent_popularity_days: int = 365
    cf_weights: tuple[float, ...] = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
    seed: int = 42


# Fallback blend weight if reports/metrics.json (which holds the tuned value) is missing.
DEFAULT_CF_WEIGHT = 0.5
