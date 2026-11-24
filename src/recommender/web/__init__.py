"""Flask web app: JSON API + single-page UI."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from ..config import DEFAULT_CF_WEIGHT, Paths
from ..engine import Engine, load_metrics
from ..tmdb import TMDBClient

MAX_RATINGS = 500
MAX_RESULTS = 30


def _bad_request(message: str):
    return jsonify({"error": message}), 400


def create_app(engine: Engine | None = None, tmdb: TMDBClient | None = None) -> Flask:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = Paths()
    engine = engine or Engine.from_artifacts(paths)
    tmdb = tmdb or TMDBClient(os.environ.get("TMDB_API_KEY"))
    metrics = load_metrics(paths.metrics)
    default_cf = float((metrics or {}).get("selected_cf_weight", DEFAULT_CF_WEIGHT))

    app = Flask(__name__)
    app.json.sort_keys = False

    def movies_json(movies) -> list[dict]:
        return tmdb.enrich([m.to_dict() for m in movies])

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            n_movies=engine.n_items,
            n_users=engine.model_card.get("n_users", 0),
            n_interactions=engine.model_card.get("n_interactions", 0),
            default_cf_weight=default_cf,
            has_sample_users=engine.sample_users is not None,
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "movies": engine.n_items, "tmdb": tmdb.enabled}

    @app.get("/api/popular")
    def popular():
        n = request.args.get("n", 24, type=int)
        seed = request.args.get("seed", type=int)
        return {"movies": movies_json(engine.popular(max(1, min(n, 60)), seed=seed))}

    @app.get("/api/search")
    def search():
        q = request.args.get("q", "")
        limit = request.args.get("limit", 8, type=int)
        return {"movies": movies_json(engine.search(q, max(1, min(limit, 20))))}

    @app.get("/api/sample-user")
    def sample_user():
        user = engine.random_sample_user(seed=request.args.get("seed", type=int))
        if user is None:
            return jsonify({"error": "No sample users available"}), 404
        top = sorted(user["ratings"].items(), key=lambda kv: -kv[1])[:8]
        history = []
        for movie_id, rating in top:
            m = engine.movie(movie_id).to_dict()
            m["rating"] = rating
            history.append(m)
        return {
            "user_idx": user["user_idx"],
            "n_ratings": len(user["ratings"]),
            "ratings": {str(k): v for k, v in user["ratings"].items()},
            "top_rated": tmdb.enrich(history),
        }

    @app.post("/api/recommend")
    def recommend():
        """Body: {"liked": [movieId, ...]} or {"ratings": {movieId: rating}}, plus
        optional "cf_weight" (0 = pure content, 1 = pure collaborative) and "n"."""
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _bad_request("Expected a JSON object")
        try:
            if "ratings" in body:
                ratings = {int(k): float(v) for k, v in dict(body["ratings"]).items()}
            else:
                ratings = Engine.liked_to_ratings([int(m) for m in body.get("liked", [])])
            cf_weight = float(body.get("cf_weight", default_cf))
            n = int(body.get("n", 12))
        except (TypeError, ValueError):
            return _bad_request("Malformed 'liked', 'ratings', 'cf_weight' or 'n'")
        if not ratings:
            return _bad_request("Provide at least one movie")
        if len(ratings) > MAX_RATINGS:
            return _bad_request(f"At most {MAX_RATINGS} ratings per request")
        if not all(0.5 <= r <= 5.0 for r in ratings.values()):
            return _bad_request("Ratings must be between 0.5 and 5")
        if not 0.0 <= cf_weight <= 1.0:
            return _bad_request("cf_weight must be between 0 and 1")
        try:
            result = engine.recommend(ratings, n=max(1, min(n, MAX_RESULTS)), cf_weight=cf_weight)
        except ValueError as e:
            return _bad_request(str(e))
        tmdb.enrich(result["recommendations"])
        return result

    @app.get("/api/model")
    def model_info():
        return {"model": engine.model_card, "metrics": metrics}

    return app
