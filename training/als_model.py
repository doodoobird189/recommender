import pandas as pd
import numpy as np
import requests
from scipy.sparse import csr_matrix, save_npz
from implicit.als import AlternatingLeastSquares

TMDB_API_KEY = "bcee530206ce4a395e6b0136cc142d8e"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

'''
Load the data
'''

print("1. Loading Data...")
df = pd.read_parquet("../data/train_data.parquet")

'''
Create the sparse matrix
    - load the data 
'''

print("2. Creating sparse matrix...")
users = df["userId_internal"].values
movies = df["movieId_internal"].values
confidence = df["confidence"].values

num_users = df["userId_internal"].max() + 1
num_movies = df["movieId_internal"].max() + 1

# R = users x movies
R = csr_matrix((confidence, (users, movies)), shape=(num_users, num_movies))

print("3. Fitting model...")

# implicit 0.7+ expects (users x items) format directly
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20
)

model.fit(R)

print("4. Saving model and sparse matrix...")
model.save("../data/als_model.npz")
save_npz("../data/R_matrix.npz", R)
print("   Saved: data/als_model.npz, data/R_matrix.npz")

print("5. Make a recommendation (smoke test)...")
user_id = 10

# pass the single user row (1, n_items)
recommended_internal_ids, scores = model.recommend(
    user_id,
    R[user_id],
    N=10
)

'''
6. Map internal IDs back to original movieIds
'''

print("6. Mapping internal IDs to movieIds...")
movie_map = pd.read_parquet("../data/movies.parquet")
# movieId_internal -> movieId
internal_to_movie = movie_map.set_index("movieId_internal")["movieId"].to_dict()

recommended_movie_ids = [internal_to_movie[i] for i in recommended_internal_ids]

'''
7. Map movieIds to TMDB IDs via links.csv
'''

print("7. Looking up TMDB IDs...")
links = pd.read_csv(
    "../data/links.csv",
    dtype={"movieId": "int32", "tmdbId": "Int32"},
    usecols=["movieId", "tmdbId"]
)
movie_to_tmdb = links.set_index("movieId")["tmdbId"].to_dict()

recommended_tmdb_ids = [movie_to_tmdb.get(mid) for mid in recommended_movie_ids]

'''
8. Fetch movie details from TMDB API
'''

def fetch_tmdb_movie(tmdb_id):
    if tmdb_id is None or pd.isna(tmdb_id):
        return None
    url = f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            "tmdb_id": tmdb_id,
            "title": data.get("title"),
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "genres": [g["name"] for g in data.get("genres", [])],
            "vote_average": data.get("vote_average"),
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
                if data.get("poster_path") else None
            ),
        }
    else:
        print(f"  Warning: TMDB returned {response.status_code} for tmdb_id={tmdb_id}")
        return None

print("8. Fetching movie details from TMDB...")
results = []
for movie_id, tmdb_id, score in zip(recommended_movie_ids, recommended_tmdb_ids, scores):
    info = fetch_tmdb_movie(tmdb_id)
    if info:
        info["score"] = float(score)
        results.append(info)

print("\n=== Recommendations for user", user_id, "===")
for i, movie in enumerate(results, 1):
    print(f"\n{i}. {movie['title']} ({movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'})")
    print(f"   Genres     : {', '.join(movie['genres'])}")
    print(f"   Rating     : {movie['vote_average']}")
    print(f"   ALS Score  : {movie['score']:.4f}")
    print(f"   Overview   : {movie['overview'][:120]}...")
    print(f"   Poster     : {movie['poster_url']}")

print("\nTraining complete. Run `python app/app.py` to start the web UI.")
