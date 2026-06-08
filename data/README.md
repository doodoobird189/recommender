# Data

This directory is not tracked by git (files are too large).

## Download

1. Go to https://grouplens.org/datasets/movielens/
2. Download **ml-latest.zip** (the full dataset)
3. Extract into this `data/` directory

You should end up with:
```
data/
├── genome-scores.csv
├── genome-tags.csv
├── links.csv
├── movies.csv
├── ratings.csv
└── tags.csv
```

Then run the preprocessing script to generate the parquet files and trained model:

```bash
cd ../pipeline && python data_cleaning.py
cd ../training  && OPENBLAS_NUM_THREADS=1 python als_model.py
```
