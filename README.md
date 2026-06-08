# Movie Recommender System

A collaborative filtering recommender system trained on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (~33M ratings, 330K users, 86K movies), served through a Flask web app with live TMDB movie data.

---

## Results

Evaluated on a held-out 20% test split across 2,000 sampled users:

| K  | Precision@K | Recall@K | NDCG@K | MAP@K  |
|----|-------------|----------|--------|--------|
| 5  | 0.4257      | 0.1333   | 0.4559 | 0.3550 |
| 10 | 0.3618      | 0.2095   | 0.4246 | 0.2916 |
| 20 | 0.2904      | 0.3096   | 0.4157 | 0.2563 |

---

## How It Works

```
ratings.csv
    │
    ▼
pipeline/data_cleaning.py
    • Filter sparse users/movies (≥20 interactions)
    • Re-encode IDs to contiguous integers
    • Build confidence scores from ratings
    │
    ▼
training/als_model.py
    • Build user×item sparse matrix (CSR)
    • Train Alternating Least Squares (implicit 0.7)
    • Save model + matrix to data/
    │
    ▼
app/app.py  ──►  GET /api/recommend?user_id=X
    • Load saved model at startup
    • Map internal IDs → movieId → tmdbId
    • Fetch poster/title/genres from TMDB API
    • Return JSON to frontend
    │
    ▼
app/templates/index.html
    • Dark-themed movie card grid
    • Poster images, genre tags, TMDB rating, ALS score
```

---

## Tech Stack

- **Model** — [implicit](https://github.com/benfred/implicit) ALS (Alternating Least Squares)
- **Data** — [MovieLens Latest](https://grouplens.org/datasets/movielens/) + [TMDB API](https://developer.themoviedb.org/)
- **Backend** — Flask
- **Frontend** — Vanilla HTML/CSS/JS
- **Data processing** — pandas, scipy sparse, pyarrow

---

## Project Structure

```
recommender/
├── app/
│   ├── app.py                 # Flask API + web server
│   └── templates/
│       └── index.html         # Frontend UI
├── data/                      # ← gitignored (see data/README.md)
├── notebooks/
│   └── explore_data.ipynb     # EDA notebook
├── pipeline/
│   └── data_cleaning.py       # Preprocessing + parquet export
├── training/
│   ├── als_model.py           # Train ALS, save model
│   └── evaluate.py            # Precision/Recall/NDCG/MAP evaluation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Get the data

Download [MovieLens Latest](https://grouplens.org/datasets/movielens/) (`ml-latest.zip`) and extract the CSVs into `data/`.

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Preprocess

```bash
cd pipeline
python data_cleaning.py
```

### 4. Train

```bash
cd training
OPENBLAS_NUM_THREADS=1 python als_model.py
```

### 5. Evaluate (optional)

```bash
cd training
OPENBLAS_NUM_THREADS=1 python evaluate.py
```

### 6. Run the web app

```bash
OPENBLAS_NUM_THREADS=1 python app/app.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## Dataset

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4):19:1–19:19.
