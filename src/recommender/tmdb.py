"""Thin, cached TMDB client for posters and overviews. Optional: the app works without a key."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org/3"
POSTER_URL = "https://image.tmdb.org/t/p/w342{path}"


class TMDBClient:
    def __init__(self, api_key: str | None, timeout: float = 4.0, max_workers: int = 8):
        self.api_key = api_key or None
        self.timeout = timeout
        self._session = requests.Session()
        self._cache: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def enabled(self) -> bool:
        return self.api_key is not None

    def details(self, tmdb_id: int | None) -> dict:
        if not self.enabled or tmdb_id is None:
            return {}
        with self._lock:
            if tmdb_id in self._cache:
                return self._cache[tmdb_id]
        try:
            resp = self._session.get(
                f"{BASE_URL}/movie/{tmdb_id}",
                params={"api_key": self.api_key},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            log.warning("TMDB request failed for %s: %s", tmdb_id, e)
            return {}  # transient: don't cache
        if resp.status_code == 404:
            info: dict = {}
        elif resp.ok:
            d = resp.json()
            info = {
                "poster_url": POSTER_URL.format(path=d["poster_path"])
                if d.get("poster_path")
                else None,
                "overview": d.get("overview") or "",
                "tmdb_rating": d.get("vote_average"),
            }
        else:
            log.warning("TMDB returned %s for %s", resp.status_code, tmdb_id)
            return {}
        with self._lock:
            self._cache[tmdb_id] = info
        return info

    def enrich(self, movies: list[dict]) -> list[dict]:
        """Add poster/overview to movie dicts (in place, fetched concurrently)."""
        if self.enabled:
            for m, info in zip(
                movies,
                self._pool.map(lambda m: self.details(m.get("tmdb_id")), movies),
                strict=True,
            ):
                m.update(info)
        return movies
