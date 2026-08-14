import os
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from scipy.sparse import load_npz
from implicit.cpu.als import AlternatingLeastSquares
from flask import Flask, jsonify, render_template, request

# ── paths (relative to this file) ──────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")

load_dotenv()

TMDB_API_KEY  = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# ── load everything once at startup ────────────────────────────────────────
print("Loading model...")
model = AlternatingLeastSquares.load(os.path.join(DATA, "als_model.npz"))

print("Loading sparse matrix...")
R = load_npz(os.path.join(DATA, "R_matrix.npz"))

print("Loading ID mappings...")
movie_map = pd.read_parquet(os.path.join(DATA, "movies.parquet"))
internal_to_movie = movie_map.set_index("movieId_internal")["movieId"].to_dict()

links = pd.read_csv(
    os.path.join(DATA, "links.csv"),
    dtype={"movieId": "int32", "tmdbId": "Int32"},
    usecols=["movieId", "tmdbId"],
)
movie_to_tmdb = links.set_index("movieId")["tmdbId"].to_dict()

NUM_USERS = R.shape[0]
print(f"Ready — {NUM_USERS:,} users in model.")

# ── TMDB helper ─────────────────────────────────────────────────────────────
def fetch_tmdb_movie(tmdb_id):
    if tmdb_id is None or pd.isna(tmdb_id):
        return None
    resp = requests.get(
        f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}",
        params={"api_key": TMDB_API_KEY},
        timeout=5,
    )
    if resp.status_code != 200:
        return None
    d = resp.json()
    return {
        "tmdb_id":      int(tmdb_id),
        "title":        d.get("title"),
        "overview":     d.get("overview", ""),
        "release_date": d.get("release_date", ""),
        "year":         (d.get("release_date") or "")[:4],
        "genres":       [g["name"] for g in d.get("genres", [])],
        "vote_average": d.get("vote_average"),
        "poster_url": (
            f"https://image.tmdb.org/t/p/w500{d['poster_path']}"
            if d.get("poster_path") else None
        ),
    }

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.template_filter('format_number')
def format_number(value):
    return f"{int(value):,}"

@app.route("/")
def index():
    return render_template("index.html", num_users=NUM_USERS)

@app.route("/api/recommend")
def recommend():
    try:
        user_id = int(request.args.get("user_id", 1))
        n       = min(int(request.args.get("n", 10)), 20)
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    if user_id < 0 or user_id >= NUM_USERS:
        return jsonify({"error": f"user_id must be between 0 and {NUM_USERS - 1}"}), 400

    ids, scores = model.recommend(user_id, R[user_id], N=n)

    results = []
    for internal_id, score in zip(ids, scores):
        movie_id = internal_to_movie.get(int(internal_id))
        if movie_id is None:
            continue
        tmdb_id = movie_to_tmdb.get(movie_id)
        info = fetch_tmdb_movie(tmdb_id)
        if info:
            info["score"] = round(float(score), 4)
            results.append(info)

    return jsonify({"user_id": user_id, "recommendations": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
