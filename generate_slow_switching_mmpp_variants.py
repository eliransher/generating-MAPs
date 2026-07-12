"""Generate 5 slow_switching_mmpp variants with different seeds.

Output: data/map_examples/slow_switching_mmpp_1.pkl .. slow_switching_mmpp_5.pkl

Moment coverage (same as the extended slow_switching_mmpp.pkl):
  Marginals : E[X^m]                for m in 1..7
  Joints    : E[X_n^i * X_{n+k}^j]
                lags 1..4  : i,j in 1..4   (w=2 for k=1, w=1 for k>=2)
                lags 5..10 : i,j in 1..2   (w=1)
Total: 7 + 64 + 24 = 95 tuples per file
"""

import sys
import pickle
from math import factorial
from pathlib import Path

import numpy as np

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

OUTPUT_DIR = Path(__file__).parent / "data" / "map_examples"
MAP_SIZE   = 10

MARGINAL_WEIGHTS = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1}

# 5 distinct seeds (offset from the original seed=_gen(12) used for the base file)
SEEDS = [101, 202, 303, 404, 505]


def compute_joint_moment(D0, D1, i, j, k):
    minus_D0 = -D0
    beta = sm.arrival_epoch_distribution(D0, D1)
    ones = np.ones(D0.shape[0])
    v = ones.copy()
    for _ in range(j):
        v = np.linalg.solve(minus_D0, v)
    for _ in range(k - 1):
        v = np.linalg.solve(minus_D0, D1 @ v)
    v = D1 @ v
    for _ in range(i + 1):
        v = np.linalg.solve(minus_D0, v)
    return float(factorial(i) * factorial(j) * (beta @ v))


def compute_moments(D0, D1):
    moments = []
    raw = sm._raw_moments(D0, D1, max_power=7)
    for m in range(1, 8):
        moments.append((m, 0, 0, raw[m], MARGINAL_WEIGHTS[m]))
    for k in range(1, 5):
        w = 2 if k == 1 else 1
        for i in range(1, 5):
            for j in range(1, 5):
                moments.append((i, j, k, compute_joint_moment(D0, D1, i, j, k), w))
    for k in range(5, 11):
        for i in range(1, 3):
            for j in range(1, 3):
                moments.append((i, j, k, compute_joint_moment(D0, D1, i, j, k), 1))
    return moments


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, seed in enumerate(SEEDS, start=1):
        name = f"slow_switching_mmpp_{idx}"
        print(f"Generating {name}  (seed={seed}) ...", end="", flush=True)
        m = sm.generate_slow_switching_mmpp(MAP_SIZE, switching_scale=0.02,
                                             rate_spread=2.5, seed=seed)
        D0, D1 = m.D0, m.D1
        mean = sm.marginal_moment(D0, D1, 1)
        moments = compute_moments(D0, D1)
        payload = {"D0": D0, "D1": D1, "moments": moments}
        out = OUTPUT_DIR / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        print(f"  mean={mean:.4f}  {len(moments)} tuples  -> {out.name}")

    print(f"\nDone. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
