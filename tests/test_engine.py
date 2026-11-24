import pytest


def test_recommend_excludes_input_and_explains(engine):
    liked = [11, 21, 31]  # movieIds for items 1, 2, 3
    res = engine.recommend(engine.liked_to_ratings(liked), n=5)
    recs = res["recommendations"]
    assert len(recs) == 5
    assert not {r["movie_id"] for r in recs} & set(liked)
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
    for r in recs:
        assert {b["movie_id"] for b in r["because_you_liked"]} <= set(liked)
        assert "great movie" not in r["shared_tags"]  # uninformative tags filtered
    assert "great movie" not in res["taste_profile"]


def test_cf_weight_changes_ranking(engine):
    ratings = engine.liked_to_ratings([11, 21, 31, 41])
    a = [r["movie_id"] for r in engine.recommend(ratings, n=10, cf_weight=1.0)["recommendations"]]
    b = [r["movie_id"] for r in engine.recommend(ratings, n=10, cf_weight=0.0)["recommendations"]]
    assert a != b


def test_unknown_movies_ignored_or_rejected(engine):
    res = engine.recommend({11: 5.0, 999_999: 5.0}, n=3)
    assert res["n_known_ratings"] == 1
    with pytest.raises(ValueError):
        engine.recommend({999_999: 5.0})


def test_search_and_popular(engine):
    assert [m.title for m in engine.search("movie 1", limit=3)][0] == "Movie 1"
    assert engine.search("x") == []
    assert engine.popular(3)[0].movie_id == 1  # item 0 has most ratings
    assert len(engine.popular(5, seed=3)) == 5


def test_sample_user(engine):
    user = engine.random_sample_user(seed=0)
    assert user["user_idx"] in (7, 9) and user["ratings"]
