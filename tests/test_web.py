import pytest

from recommender.tmdb import TMDBClient
from recommender.web import create_app


@pytest.fixture
def client(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("RECSYS_REPORTS_DIR", str(tmp_path))
    app = create_app(engine=engine, tmdb=TMDBClient(api_key=None))
    app.testing = True
    return app.test_client()


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/healthz").get_json()["status"] == "ok"


def test_recommend_liked(client):
    resp = client.post("/api/recommend", json={"liked": [11, 21], "n": 4, "cf_weight": 0.5})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["recommendations"]) == 4 and body["cf_weight"] == 0.5


def test_recommend_ratings(client):
    resp = client.post("/api/recommend", json={"ratings": {"11": 5, "21": 1.0}})
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"liked": []},
        {"liked": ["abc"]},
        {"liked": [11], "cf_weight": 2},
        {"ratings": {"11": 9}},
        {"liked": [123456789]},
    ],
)
def test_recommend_validation(client, payload):
    assert client.post("/api/recommend", json=payload).status_code == 400


def test_search_popular_sample(client):
    assert client.get("/api/search?q=movie").get_json()["movies"]
    assert len(client.get("/api/popular?n=5").get_json()["movies"]) == 5
    sample = client.get("/api/sample-user?seed=1").get_json()
    assert sample["ratings"] and sample["top_rated"]
