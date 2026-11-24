import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recommender.scoring import ALSScorer, ContentScorer, blend, top_k


def test_fold_in_matches_implicit(item_factors):
    implicit = pytest.importorskip("implicit.cpu.als")
    model = implicit.AlternatingLeastSquares(
        factors=item_factors.shape[1], regularization=0.1, use_cg=False
    )
    model.item_factors = item_factors
    items, conf = np.array([1, 5, 9]), np.array([2.0, 6.0, 4.5], dtype=np.float32)
    row = csr_matrix((conf, (np.zeros(3, int), items)), shape=(1, len(item_factors)))
    expected = model.recalculate_user(0, row)
    ours = ALSScorer(item_factors, 0.1).fold_in(items, conf).x
    np.testing.assert_allclose(ours, np.ravel(expected), rtol=1e-4, atol=1e-5)


def test_contributions_sum_to_score(item_factors):
    als = ALSScorer(item_factors, 0.1)
    fold = als.fold_in(np.array([0, 3, 7]), np.array([6.0, 2.0, 4.0]))
    targets = np.array([10, 20, 30])
    contrib = als.contributions(fold, targets)
    assert contrib.shape == (3, 3)
    np.testing.assert_allclose(contrib.sum(axis=1), als.scores(fold.x)[targets], rtol=1e-4)


def test_content_profile_prefers_similar_items(genome):
    content = ContentScorer(genome)
    liked = np.array([4])
    p = content.profile(liked, np.array([5.0]))
    assert p is not None and np.isclose(np.linalg.norm(p), 1)
    scores = content.scores(p)
    assert np.argmax(scores) == 4  # most similar to itself
    assert (scores[~genome.has] == 0).all()


def test_content_profile_none_without_genome(genome):
    assert ContentScorer(genome).profile(np.array([35, 36]), np.array([5.0, 4.0])) is None


def test_disliked_items_push_away(genome):
    content = ContentScorer(genome)
    p = content.profile(np.array([4]), np.array([0.5]))
    assert np.argmin(content.scores(p)) == 4


def test_top_k_excludes_and_sorts():
    s = np.array([[0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.2, 0.3]])
    ex = np.array([[False, True, False, False], [False, False, False, False]])
    np.testing.assert_array_equal(top_k(s, 2, ex), [[3, 2], [0, 3]])
    np.testing.assert_array_equal(top_k(s[0], 3), [1, 3, 2])


def test_blend_extremes_reduce_to_single_signal(genome):
    rng = np.random.default_rng(0)
    als, content = rng.normal(size=40), rng.normal(size=40) * genome.has
    exclude = np.zeros(40, bool)
    exclude[:3] = True
    pure_cf = top_k(blend(als, content, 1.0, genome.has, exclude), 5, exclude)
    np.testing.assert_array_equal(pure_cf, top_k(als, 5, exclude))
    pure_content = blend(als, content, 0.0, genome.has, exclude)
    assert (pure_content[~genome.has] == 0).all()  # no genome -> neutral, not penalised


def test_scale_to_top_anchors_head_at_one():
    from recommender.scoring import _scale_to_top

    rng = np.random.default_rng(0)
    heavy, light = rng.pareto(1.5, size=(1, 500)), rng.normal(size=(1, 500))
    valid = np.ones((1, 500), bool)
    for s in (heavy, light):
        n = _scale_to_top(s, valid, head=10)
        assert np.mean(np.sort(n[0])[-10:]) == pytest.approx(1.0, abs=1e-5)
        assert np.mean(n) == pytest.approx(0.0, abs=1e-5)


def test_blend_lets_content_surface_cold_items(genome):
    # item 5: cold for CF (score ~ catalogue mean), but the best content match
    als = np.linspace(0, 1, 40) ** 8  # heavy-tailed, like real ALS scores
    als[5] = als.mean()
    content = np.where(genome.has, 0.0, 0.0)
    content[5] = 1.0
    exclude = np.zeros(40, bool)
    top = top_k(blend(als, content, 0.4, genome.has, exclude), 3, exclude)
    assert 5 in top
