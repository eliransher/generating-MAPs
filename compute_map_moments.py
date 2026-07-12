"""Generate one MMPP (original method, size 5, mean 1) and compute its moment
statistics for use as a fitting target in gradient-descent MAP fitting.

Output: data/target_map.pkl
  {
    'D0'     : ndarray (5, 5),
    'D1'     : ndarray (5, 5),
    'moments': list of (i, j, k, value, w_L) tuples
  }

Tuple conventions
-----------------
Marginal moments  E[X^m]          -> (m, 0, 0, value, w_L)
Joint moments     E[X_n^i*X_{n+k}^j] -> (i, j, k, value, w_L)

Weights w_L
-----------
Marginals : m=1->10, m=2->5, m=3->3, m=4->2, m=5->2
Joints    : k=1->2, k=2 or k=3->1

Total tuples: 5 marginals + 27 joints (i,j,k in {1,2,3}) = 32
"""

import sys
import pickle
from math import factorial
from pathlib import Path

import numpy as np

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

OUTPUT_PATH = Path(__file__).parent / "data" / "target_map.pkl"
MAP_SIZE    = 5
SEED        = 42

MARGINAL_WEIGHTS = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1}


# ---------------------------------------------------------------------------
# Joint moment computation
# ---------------------------------------------------------------------------

def compute_joint_moment(D0: np.ndarray, D1: np.ndarray,
                         i: int, j: int, k: int) -> float:
    """Compute E[X_n^i * X_{n+k}^j] for a MAP in stationarity.

    Derivation
    ----------
    Let beta  = arrival-epoch distribution (row, n),
        K     = (-D0)^{-1} D1  (embedded transition matrix, n x n).

    E[X_n^i * X_{n+k}^j]
      = i! * j! * beta @ (-D0)^{-(i+1)} @ D1 @ K^{k-1} @ (-D0)^{-j} @ 1

    Verified against `_first_lag_correlation_from_moments` for i=j=k=1.

    Parameters
    ----------
    i, j : positive int  — moment orders for X_n and X_{n+k}
    k    : positive int  — lag between the two inter-arrivals
    """
    minus_D0 = -D0
    beta     = sm.arrival_epoch_distribution(D0, D1)
    ones     = np.ones(D0.shape[0])

    # ---- build result vector right-to-left ----

    # 1. v = (-D0)^{-j} @ 1   (j linear solves)
    v = ones.copy()
    for _ in range(j):
        v = np.linalg.solve(minus_D0, v)

    # 2. Apply K^{k-1} from the left:  K = (-D0)^{-1} @ D1
    #    Each step: v <- K @ v = solve(-D0, D1 @ v)
    for _ in range(k - 1):
        v = np.linalg.solve(minus_D0, D1 @ v)

    # 3. Apply D1 from the left
    v = D1 @ v

    # 4. Apply (-D0)^{-(i+1)} from the left  (i+1 solves)
    for _ in range(i + 1):
        v = np.linalg.solve(minus_D0, v)

    # 5. Contract with beta and scale by factorials
    return float(factorial(i) * factorial(j) * (beta @ v))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- generate MMPP (original method, size 5, mean 1) ---
    mmpp = sm.generate_random_mmpp(MAP_SIZE, seed=SEED)
    D0, D1 = mmpp.D0, mmpp.D1

    mean_check = sm.marginal_moment(D0, D1, 1)
    print(f"Generated MMPP  size={MAP_SIZE}  seed={SEED}")
    print(f"  E[X^1] = {mean_check:.8f}  (should be 1.0)")
    print(f"\nD0 =\n{D0}")
    print(f"\nD1 =\n{D1}")

    moments_list: list[tuple] = []

    # --- marginal moments E[X^m] for m = 1 .. 5 ---
    raw = sm._raw_moments(D0, D1, max_power=7)   # raw[m] = E[X^m]

    print("\n--- Marginal moments ---")
    for m in range(1, 8):
        val = raw[m]
        w   = MARGINAL_WEIGHTS[m]
        moments_list.append((m, 0, 0, val, w))
        print(f"  E[X^{m}]                    = {val:.8f}   w={w}")

    # --- joint moments E[X_n^i * X_{n+k}^j] for i,j,k in {1,2,3} ---
    print("\n--- Joint moments ---")
    for k in range(1, 4):
        w = 2 if k == 1 else 1
        for i in range(1, 4):
            for j in range(1, 4):
                val = compute_joint_moment(D0, D1, i, j, k)
                moments_list.append((i, j, k, val, w))
                print(f"  E[X_n^{i} * X_{{n+{k}}}^{j}]     = {val:.8f}   w={w}")

    # --- pickle ---
    payload = {"D0": D0, "D1": D1, "moments": moments_list}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nTotal tuples : {len(moments_list)}  (7 marginals + 27 joints)")
    print(f"Saved        -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
