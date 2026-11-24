"""ALS training (requires the `train` dependency group: implicit + threadpoolctl)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix

from .config import ALSConfig

log = logging.getLogger(__name__)


def train_als(conf: csr_matrix, cfg: ALSConfig, show_progress: bool = True) -> np.ndarray:
    """Fit implicit-feedback ALS on a user x item confidence matrix; return the item factors."""
    try:
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Training needs the 'train' dependency group: `uv sync --group train`"
        ) from e

    model = AlternatingLeastSquares(
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        alpha=1.0,  # confidence (1 + alpha * rating) is computed up front, see data.confidence
        random_state=cfg.seed,
    )
    log.info("Training ALS on %s users x %s items (%s nnz)", *conf.shape, f"{conf.nnz:,}")
    # implicit parallelises internally; nested BLAS threading only slows it down
    with threadpool_limits(1, "blas"):
        model.fit(conf, show_progress=show_progress)
    return np.asarray(model.item_factors, dtype=np.float32)
