import numpy as np
import pytest

from recommender.evaluation import intra_list_diversity, markdown_table, ranking_metrics


def test_ranking_metrics_perfect():
    m = ranking_metrics(np.array([1, 2, 3]), {1, 2, 3}, k=3)
    assert m == pytest.approx({"precision": 1, "recall": 1, "ndcg": 1, "map": 1})


def test_ranking_metrics_hand_computed():
    # hits at ranks 1 and 3, 4 relevant items overall
    m = ranking_metrics(np.array([10, 99, 20, 98]), {10, 20, 30, 40}, k=4)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["map"] == pytest.approx((1 / 1 + 2 / 3) / 4)
    idcg = sum(1 / np.log2(i + 2) for i in range(4))
    assert m["ndcg"] == pytest.approx((1 + 1 / np.log2(4)) / idcg)


def test_ranking_metrics_no_hits():
    assert ranking_metrics(np.array([5, 6]), {1}, k=2)["ndcg"] == 0


def test_diversity_bounds(genome):
    assert intra_list_diversity(np.array([3, 3]), genome) == pytest.approx(0, abs=1e-6)
    assert intra_list_diversity(np.array([35, 36]), genome) is None
    d = intra_list_diversity(np.arange(10), genome)
    assert 0 < d < 2


def test_markdown_table_bolds_best():
    table = markdown_table(
        {"A": {"precision@10": 0.1}, "B": {"precision@10": 0.3}}, include_cold=False
    )
    assert "**0.3000**" in table and "| A | 0.1000" in table


def test_select_cf_weight_considers_endpoints():
    from recommender.evaluation import select_cf_weight

    models = {
        "Content (genome)": {"ndcg@10": 0.1},
        "ALS": {"ndcg@10": 0.3},
        "Hybrid (cf=0.5)": {"ndcg@10": 0.25},
    }
    assert select_cf_weight(models, (0.5,)) == 1.0
    models["Hybrid (cf=0.5)"]["ndcg@10"] = 0.4
    assert select_cf_weight(models, (0.5,)) == 0.5
