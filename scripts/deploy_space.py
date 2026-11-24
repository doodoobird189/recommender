"""Deploy the web app to a Hugging Face Docker Space.

The Space builds the same Dockerfile as local Docker. Trained artifacts are not in git, so this
script stages exactly what the image needs (source, lockfile, artifacts, reports) plus a Space
README carrying the required config header, then uploads it in a single commit.

Usage:  uv run python scripts/deploy_space.py [--space user/name]
Auth:   `hf auth login` with a token that has write access to the Space.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = "doodoobird189/movie-recommender"
GITHUB_URL = "https://github.com/doodoobird189/recommender"

FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", ".python-version"]
DIRS = ["src", "artifacts", "reports"]
REQUIRED_ARTIFACTS = [
    "items.parquet",
    "genome.npz",
    "als_item_factors.npy",
    "model.json",
    "sample_users.parquet",
]

SPACE_README = """---
title: Movie Recommender
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: Hybrid ALS + tag-genome movie recommender
---

# Hybrid Movie Recommender

Pick a few movies you love and get explained recommendations from a hybrid of collaborative
filtering (implicit ALS on 32M MovieLens ratings) and the MovieLens tag genome.

Source code, evaluation and write-up: {github}

_Deployed from commit `{commit}`._
"""


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def stage(dest: Path, commit: str) -> None:
    missing = [a for a in REQUIRED_ARTIFACTS if not (ROOT / "artifacts" / a).exists()]
    if missing:
        sys.exit(f"Missing artifacts {missing}: run `make pipeline` first.")
    for f in FILES:
        shutil.copy2(ROOT / f, dest / f)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep")
    for d in DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=ignore)
    (dest / "README.md").write_text(SPACE_README.format(github=GITHUB_URL, commit=commit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=DEFAULT_SPACE, help="Space repo id (user/name)")
    args = parser.parse_args()

    commit = git_commit()
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage(staging, commit)
        size_mb = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()) / 1e6
        print(f"Uploading {size_mb:.1f} MB to spaces/{args.space} (commit {commit})...")
        info = HfApi().upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=staging,
            commit_message=f"Deploy {commit}",
            # remove files that no longer exist locally, but never touch .gitattributes
            delete_patterns=["src/**", "artifacts/**", "reports/**"],
        )
    print(f"Uploaded: {info.commit_url}")
    owner, name = args.space.split("/")
    print(f"Build logs: https://huggingface.co/spaces/{args.space}?logs=build")
    print("App:        " + f"https://{owner}-{name}.hf.space".replace("_", "-").lower())


if __name__ == "__main__":
    main()
