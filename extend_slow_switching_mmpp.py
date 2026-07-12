"""Extend data/map_examples/slow_switching_mmpp.pkl with 4th-order and 4th-lag moments.

New moment coverage
-------------------
Marginals : E[X^m]               for m in 1..7   (unchanged)
Joints    : E[X_n^i * X_{n+k}^j] for i,j in 1..4, k in 1..4

Weights
-------
Marginals : {1:10, 2:5, 3:3, 4:2, 5:2, 6:1, 7:1}  (unchanged)
Joints    : k=1 -> 2,  k>=2 -> 1

Total tuples: 7 marginals + 4*4*4 = 64 joints = 71

Usage
-----
    python extend_slow_switching_mmpp.py
"""

import sys
import pickle
from math import factorial
from pathlib import Path

import numpy as np

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

INPUT_PATH  = Path(__file__).parent / "data" / "map_examples" / "slow_switching_mmpp.pkl"
OUTPUT_PATH = INPUT_PATH  # overwrite in place

MAX_ORDER       = 4   # i, j up to this value for lags 1..4
MAX_LAG         = 4   # full-order lags
MAX_LAG_REDUCED = 10  # lags 5..10 use reduced order
MAX_ORDER_HIGH_LAG = 2  # i, j up to this value for lags 5+

MARGINAL_WEIGHTS = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1}


def compute_joint_moment(D0: np.ndarray, D1: np.ndarray,
                         i: int, j: int, k: int) -> float:
    """E[X_n^i * X_{n+k}^j] = i! * j! * beta @ (-D0)^{-(i+1)} @ D1 @ K^{k-1} @ (-D0)^{-j} @ 1"""
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


def compute_extended_moments(D0: np.ndarray, D1: np.ndarray) -> list[tuple]:
    moments: list[tuple] = []

    # marginal moments E[X^m]
    raw = sm._raw_moments(D0, D1, max_power=7)
    for m in range(1, 8):
        moments.append((m, 0, 0, raw[m], MARGINAL_WEIGHTS[m]))

    # joint moments: lags 1..4, full order (i,j up to 4)
    for k in range(1, MAX_LAG + 1):
        w = 2 if k == 1 else 1
        for i in range(1, MAX_ORDER + 1):
            for j in range(1, MAX_ORDER + 1):
                val = compute_joint_moment(D0, D1, i, j, k)
                moments.append((i, j, k, val, w))

    # joint moments: lags 5..10, reduced order (i,j up to 2)
    for k in range(MAX_LAG + 1, MAX_LAG_REDUCED + 1):
        for i in range(1, MAX_ORDER_HIGH_LAG + 1):
            for j in range(1, MAX_ORDER_HIGH_LAG + 1):
                val = compute_joint_moment(D0, D1, i, j, k)
                moments.append((i, j, k, val, 1))

    return moments


def main() -> None:
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)

    D0, D1 = data["D0"], data["D1"]
    print(f"Loaded {INPUT_PATH.name}  (MAP size {D0.shape[0]}x{D0.shape[0]})")
    print(f"Old moment count : {len(data['moments'])}")

    moments = compute_extended_moments(D0, D1)
    n_joints_full    = MAX_ORDER**2 * MAX_LAG
    n_joints_reduced = MAX_ORDER_HIGH_LAG**2 * (MAX_LAG_REDUCED - MAX_LAG)
    print(f"New moment count : {len(moments)}  "
          f"(7 marginals + {n_joints_full} full-order joints + {n_joints_reduced} reduced-order joints)")

    print("\n--- New / extended tuples (orders or lags >= 4) ---")
    for t in moments:
        i, j, k, val, w = t
        if i == 4 or j == 4 or k == 4:
            print(f"  ({i},{j},lag={k})  val={val:.6g}  w={w}")

    payload = {"D0": D0, "D1": D1, "moments": moments}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
