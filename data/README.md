# Data

Nothing in this directory is tracked by git except this file.

## Download

```bash
make download        # fetches ml-latest.zip (~335 MB) from GroupLens and unzips to data/ml-latest/
```

Or manually: download **ml-latest.zip** from <https://grouplens.org/datasets/movielens/> and
extract it here, so that you have:

```
data/
└── ml-latest/
    ├── genome-scores.csv   # 18.5M (movie, tag, relevance) rows — the tag genome
    ├── genome-tags.csv     # 1,128 tag names
    ├── links.csv           # movieId -> imdbId / tmdbId
    ├── movies.csv          # titles + genres
    ├── ratings.csv         # 33.8M ratings, 1995–2023
    └── tags.csv            # free-text user tags (unused)
```

## Build

```bash
make pipeline        # = uv run recommender prepare && train && evaluate
```

| Step | Writes | Notes |
|---|---|---|
| `prepare` | `data/processed/interactions.parquet`, `artifacts/{items.parquet,genome.npz,sample_users.parquet}` | 20-core filtering, contiguous ids, genome aligned to the catalogue |
| `train` | `artifacts/{als_item_factors.npy,model.json}` | ALS on all interactions (~1–2 min on 12 cores) |
| `evaluate` | `reports/{metrics.json,metrics.md}` | trains 2 more models (random + temporal split) |

The data is licensed by GroupLens for non-commercial use. See `ml-latest/README.txt`.
