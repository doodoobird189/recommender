"""Command-line entry point:  uv run recommender {prepare,train,evaluate,serve,all}."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import ALSConfig, DataConfig, EvalConfig, Paths

log = logging.getLogger("recommender")


def prepare(paths: Paths, cfg: DataConfig) -> None:
    """Raw MovieLens CSVs -> cleaned interactions + serving metadata (items, genome, samples)."""
    from .data import build_items_table, encode_ids, filter_k_core, load_movies, load_ratings
    from .genome import build_genome

    if not (paths.raw / "ratings.csv").exists():
        raise SystemExit(f"{paths.raw / 'ratings.csv'} not found — see data/README.md")
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    log.info("Loading ratings from %s", paths.raw)
    ratings = load_ratings(paths.raw)
    log.info("Loaded %s ratings", f"{len(ratings):,}")
    ratings = filter_k_core(ratings, cfg.min_user_interactions, cfg.min_item_interactions)
    df, users, items_index = encode_ids(ratings)
    log.info(
        "After %d-core filtering: %s ratings, %s users, %s movies",
        cfg.min_user_interactions,
        f"{len(df):,}",
        f"{len(users):,}",
        f"{len(items_index):,}",
    )
    df[["user_idx", "item_idx", "rating", "timestamp"]].to_parquet(paths.interactions, index=False)

    items = build_items_table(items_index, df, load_movies(paths.raw))
    items.to_parquet(paths.items, index=False)

    log.info("Building tag genome matrix...")
    genome = build_genome(paths.raw, items["movieId"].to_numpy())
    np.savez_compressed(paths.genome, **genome)
    log.info("Genome covers %s / %s movies", f"{len(genome['item_idx']):,}", f"{len(items):,}")

    # a handful of real users (moderate history length) for the "random MovieLens user" demo
    counts = df.groupby("user_idx").size()
    pool = counts[(counts >= 30) & (counts <= 300)].index.to_numpy()
    chosen = np.random.default_rng(0).choice(
        pool, size=min(cfg.n_sample_users, len(pool)), replace=False
    )
    df.loc[df["user_idx"].isin(chosen), ["user_idx", "item_idx", "rating"]].to_parquet(
        paths.sample_users, index=False
    )
    log.info("Wrote artifacts to %s", paths.artifacts)


def _load_interactions(paths: Paths) -> pd.DataFrame:
    if not paths.interactions.exists():
        raise SystemExit("No processed data — run `uv run recommender prepare` first")
    return pd.read_parquet(paths.interactions)


def train(paths: Paths, cfg: ALSConfig) -> None:
    """Fit ALS on all interactions and write the serving model."""
    from .data import confidence, to_csr
    from .train import train_als

    df = _load_interactions(paths)
    n_users, n_items = int(df["user_idx"].max()) + 1, int(df["item_idx"].max()) + 1
    conf = to_csr(df, n_users, n_items, confidence(df["rating"].to_numpy(), cfg.alpha))
    start = time.perf_counter()
    item_factors = train_als(conf, cfg)
    elapsed = time.perf_counter() - start

    np.save(paths.item_factors, item_factors)
    card = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_seconds": round(elapsed, 1),
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(len(df)),
        "als": cfg.to_dict(),
    }
    paths.model_card.write_text(json.dumps(card, indent=2))
    log.info("Saved %s (%s) in %.0fs", paths.item_factors, item_factors.shape, elapsed)


def evaluate(paths: Paths, als_cfg: ALSConfig, eval_cfg: EvalConfig) -> None:
    """Benchmark all models under a random and a temporal split; write reports/."""
    from .data import random_split, temporal_split
    from .evaluation import evaluate_split, markdown_table, select_cf_weight
    from .genome import Genome
    from .train import train_als

    df = _load_interactions(paths)
    n_users, n_items = int(df["user_idx"].max()) + 1, int(df["item_idx"].max()) + 1
    genome = Genome.load(paths.genome, n_items)
    paths.reports.mkdir(parents=True, exist_ok=True)

    def run(train_df, test_df, **kw):
        return evaluate_split(
            train_df,
            test_df,
            n_users,
            n_items,
            genome,
            als_cfg,
            eval_cfg,
            lambda c, a: train_als(c, a, show_progress=False),
            **kw,
        )

    tr, te, cutoff = temporal_split(df, eval_cfg.test_fraction)

    # Tune the blend weight on a validation window that ends where the test window begins, so
    # the test numbers below are never used to choose anything.
    log.info("=== Validation window (tunes the hybrid weight; test data untouched) ===")
    vtr, vte, val_cutoff = temporal_split(tr, eval_cfg.test_fraction / (1 - eval_cfg.test_fraction))
    val_res = run(vtr, vte)
    selected = select_cf_weight(val_res["models"], eval_cfg.cf_weights)
    log.info("Selected cf_weight=%s on validation NDCG@10", selected)

    log.info(
        "=== Temporal test split (last %.0f%% of interactions by time) ===",
        eval_cfg.test_fraction * 100,
    )
    temporal_res = run(tr, te, recent_since=cutoff - eval_cfg.recent_popularity_days * 86_400)
    temporal_res["cutoff"] = datetime.fromtimestamp(cutoff, UTC).date().isoformat()

    log.info("=== Random split (%.0f%% of interactions held out) ===", eval_cfg.test_fraction * 100)
    random_res = run(*random_split(df, eval_cfg.test_fraction, eval_cfg.seed))

    validation = {
        "window": [
            datetime.fromtimestamp(val_cutoff, UTC).date().isoformat(),
            temporal_res["cutoff"],
        ],
        "ndcg@10": {name: m["ndcg@10"] for name, m in val_res["models"].items()},
    }

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": int(len(df)),
            "genome_coverage": round(float(genome.has.mean()), 4),
        },
        "als": als_cfg.to_dict(),
        "eval": {
            "test_fraction": eval_cfg.test_fraction,
            "n_eval_users": eval_cfg.n_eval_users,
            "cold_item_threshold": eval_cfg.cold_item_threshold,
            "seed": eval_cfg.seed,
        },
        "selected_cf_weight": selected,
        "validation": validation,
        "splits": {"random": random_res, "temporal": temporal_res},
    }
    paths.metrics.write_text(json.dumps(report, indent=2))

    md = [
        "# Evaluation results",
        "",
        f"_Generated {report['generated_at']} — {eval_cfg.n_eval_users:,} sampled users per "
        "split. All metrics: higher is better._",
        "",
        "## Temporal split (train on the past, predict the future)",
        "",
        f"Cutoff **{temporal_res['cutoff']}**. "
        f"{temporal_res['n_users_with_cold_test_items']:,} evaluated users interacted with a cold "
        f"movie (< {eval_cfg.cold_item_threshold} training ratings) after the cutoff.",
        "",
        markdown_table(temporal_res["models"]),
        "",
        f"The app serves **Hybrid (cf={selected})**: the weight with the best NDCG@10 on the "
        f"validation window {validation['window'][0]} → {validation['window'][1]}, chosen before "
        "looking at the test window above.",
        "",
        "## Random split",
        "",
        markdown_table(random_res["models"], include_cold=False),
        "",
    ]
    (paths.reports / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))


def serve(host: str, port: int, debug: bool) -> None:
    from .web import create_app

    create_app().run(host=host, port=port, debug=debug)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recommender", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="clean raw MovieLens data and build artifacts")
    sub.add_parser("train", help="train the serving ALS model")
    ev = sub.add_parser("evaluate", help="benchmark models, write reports/")
    ev.add_argument("--users", type=int, default=EvalConfig.n_eval_users)
    sub.add_parser("all", help="prepare + train + evaluate")
    sv = sub.add_parser("serve", help="run the development web server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    sv.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    paths, als_cfg = Paths(), ALSConfig()

    if args.command in ("prepare", "all"):
        prepare(paths, DataConfig())
    if args.command in ("train", "all"):
        train(paths, als_cfg)
    if args.command == "evaluate":
        evaluate(paths, als_cfg, EvalConfig(n_eval_users=args.users))
    if args.command == "all":
        evaluate(paths, als_cfg, EvalConfig())
    if args.command == "serve":
        serve(args.host, args.port, args.debug)


if __name__ == "__main__":
    main()
