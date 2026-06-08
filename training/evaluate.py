import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split

# ── config ──────────────────────────────────────────────────────────────────
FACTORS        = 64
REGULARIZATION = 0.1
ITERATIONS     = 20
TEST_FRACTION  = 0.2   # fraction of each user's ratings held out for testing
K_VALUES       = [5, 10, 20]
EVAL_USERS     = 2000  # users to sample for evaluation (full set is slow)
SEED           = 42

# ── metric helpers ───────────────────────────────────────────────────────────
def evaluate(model, R_train, R_test, k, n_users=EVAL_USERS, seed=SEED):
    """
    Compute Precision, Recall, NDCG, and MAP at K over a random sample of users.
    Only evaluates users who have at least 1 held-out test item.
    """
    rng = np.random.default_rng(seed)
    all_users = np.where(np.diff(R_test.indptr) > 0)[0]  # users with test items
    sampled   = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)

    precisions, recalls, ndcgs, aps = [], [], [], []

    for uid in sampled:
        # ground-truth: items this user had in the test set
        test_items = set(R_test[uid].indices)
        if not test_items:
            continue

        # top-K recommendations (excluding already-seen training items)
        rec_ids, _ = model.recommend(uid, R_train[uid], N=k)

        hits = [1 if item in test_items else 0 for item in rec_ids]
        n_hits = sum(hits)

        # Precision@K
        precisions.append(n_hits / k)

        # Recall@K
        recalls.append(n_hits / len(test_items))

        # NDCG@K
        ideal_hits = min(len(test_items), k)
        ideal_dcg  = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        dcg        = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

        # MAP@K (average precision)
        ap, running = 0.0, 0
        for i, h in enumerate(hits):
            if h:
                running += 1
                ap += running / (i + 1)
        aps.append(ap / min(len(test_items), k))

    return {
        "Precision": np.mean(precisions),
        "Recall":    np.mean(recalls),
        "NDCG":      np.mean(ndcgs),
        "MAP":       np.mean(aps),
        "n_users":   len(precisions),
    }

# ── load data ────────────────────────────────────────────────────────────────
print("1. Loading data...")
df = pd.read_parquet("../data/train_data.parquet")

users      = df["userId_internal"].values
movies     = df["movieId_internal"].values
confidence = df["confidence"].values

num_users  = int(df["userId_internal"].max()) + 1
num_movies = int(df["movieId_internal"].max()) + 1

R = csr_matrix((confidence, (users, movies)), shape=(num_users, num_movies))
print(f"   Matrix: {num_users:,} users × {num_movies:,} movies  ({R.nnz:,} ratings)")

# ── train / test split ───────────────────────────────────────────────────────
print(f"\n2. Splitting (train {int((1-TEST_FRACTION)*100)}% / test {int(TEST_FRACTION*100)}%)...")
R_train, R_test = train_test_split(R, train_percentage=1 - TEST_FRACTION)
print(f"   Train ratings: {R_train.nnz:,}   Test ratings: {R_test.nnz:,}")

# ── train ALS ────────────────────────────────────────────────────────────────
print(f"\n3. Training ALS (factors={FACTORS}, reg={REGULARIZATION}, iters={ITERATIONS})...")
model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
)
model.fit(R_train)

# ── evaluate at each K ───────────────────────────────────────────────────────
print(f"\n4. Evaluating on {EVAL_USERS:,} sampled users...\n")

header  = f"{'K':>4}  {'Precision':>10}  {'Recall':>10}  {'NDCG':>10}  {'MAP':>10}  {'Users':>8}"
divider = "─" * len(header)
print(header)
print(divider)

for k in K_VALUES:
    m = evaluate(model, R_train, R_test, k=k)
    print(
        f"{k:>4}  "
        f"{m['Precision']:>10.4f}  "
        f"{m['Recall']:>10.4f}  "
        f"{m['NDCG']:>10.4f}  "
        f"{m['MAP']:>10.4f}  "
        f"{m['n_users']:>8,}"
    )

print(divider)
print("\nMetric definitions:")
print("  Precision@K — of the top-K recs, fraction the user actually rated")
print("  Recall@K    — fraction of the user's test items that appear in top-K")
print("  NDCG@K      — rewards ranking relevant items higher in the list (0–1)")
print("  MAP@K       — mean average precision across sampled users (0–1)")

