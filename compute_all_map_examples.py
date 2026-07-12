"""Generate one target MAP per process type and compute its moment statistics.

Each output pickle has the same format as target_map.pkl:
    {
        'D0'     : ndarray (n, n),
        'D1'     : ndarray (n, n),
        'moments': list of (i, j, k, value, w_L) tuples
    }

Tuple conventions
-----------------
Marginal moments  E[X^m]                -> (m, 0, 0, value, w_L)
Joint moments     E[X_n^i * X_{n+k}^j] -> (i, j, k, value, w_L)

Weights
-------
Marginals : m=1->10, m=2->5, m=3->3, m=4->2, m=5->2, m=6->1, m=7->1
Joints    : k=1->2, k=2 or k=3->1

Total tuples per file: 7 + 27 = 34

Output: data/map_examples/<process_type>.pkl
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
SEED       = 42

MARGINAL_WEIGHTS = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1}


# ---------------------------------------------------------------------------
# Moment computation (same as compute_map_moments.py)
# ---------------------------------------------------------------------------

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


def compute_moments_list(D0: np.ndarray, D1: np.ndarray) -> list[tuple]:
    moments: list[tuple] = []
    raw = sm._raw_moments(D0, D1, max_power=7)
    for m in range(1, 8):
        moments.append((m, 0, 0, raw[m], MARGINAL_WEIGHTS[m]))
    for k in range(1, 4):
        w = 2 if k == 1 else 1
        for i in range(1, 4):
            for j in range(1, 4):
                val = compute_joint_moment(D0, D1, i, j, k)
                moments.append((i, j, k, val, w))
    return moments


# ---------------------------------------------------------------------------
# One generator per MAP type (size ~MAP_SIZE, seed-based)
# ---------------------------------------------------------------------------

def _gen(seed_offset: int):
    return int(np.random.default_rng(SEED + seed_offset).integers(0, 2**31))


def generate_map(process_type: str) -> sm.MAP:
    n  = MAP_SIZE
    na = n // 2
    nb = n - na
    rng = np.random.default_rng(SEED)

    if process_type == "mmpp":
        return sm.generate_random_mmpp(n, seed=_gen(0))

    if process_type == "alternating_blocks":
        return sm.generate_alternating_blocks_map(na, nb, "sparse", "sparse",
                                                   seed=_gen(1), entry_mode="first_phase")

    if process_type == "alternating_extreme":
        return sm.generate_alternating_extreme_map(na, nb, seed=_gen(2))

    if process_type == "alternating_extreme_hyper":
        return sm.generate_alternating_extreme_hyper_map(na, nb, seed=_gen(3))

    if process_type == "diverse_ph":
        return sm.generate_diverse_ph_map("coxian_ph", n, seed=_gen(4))

    if process_type == "modified_renewal_ph":
        return sm.generate_modified_renewal_ph_map("coxian_ph", n, seed=_gen(5),
                                                    correlation_sign=1)

    if process_type == "random_map":
        return sm.generate_random_map_method_map(n, seed=_gen(6))

    if process_type == "random_map_negative1":
        return sm.generate_random_map_negative1_method_map(n, seed=_gen(7))

    if process_type == "erlang_renewal":
        return sm.generate_erlang_renewal_map(n, seed=_gen(8))

    if process_type == "sparse_ph_renewal":
        return sm.generate_sparse_ph_renewal_map(n, seed=_gen(9))

    if process_type == "hyper_erlang_renewal":
        return sm.generate_hyper_erlang_renewal_map(n, seed=_gen(10))

    if process_type == "modified_hyper_erlang_renewal":
        return sm.generate_modified_hyper_erlang_renewal_map(n, seed=_gen(11))

    if process_type == "slow_switching_mmpp":
        return sm.generate_slow_switching_mmpp(n, switching_scale=0.02,
                                                rate_spread=2.5, seed=_gen(12))

    if process_type == "erlang_modulated":
        return sm.generate_erlang_modulated_map(2, na, same_mode_concentration=8.0,
                                                 seed=_gen(13))

    if process_type == "hyper_telegraph":
        return sm.generate_hyper_telegraph_map(na, nb, seed=_gen(14))

    if process_type == "erlang_telegraph":
        return sm.generate_erlang_telegraph_map(na, nb, rate_ratio=2.0, seed=_gen(15))

    if process_type == "erlang_rate_spread":
        return sm.generate_erlang_rate_spread_map(2, na, rate_spread=200.0,
                                                   same_mode_concentration=10.0,
                                                   seed=_gen(16))

    if process_type == "coxian_alternating":
        return sm.generate_coxian_alternating_map(na, nb, rate_ratio=5.0,
                                                   exit_shape=1.5, seed=_gen(17))

    if process_type == "erlang_hyper_telegraph":
        return sm.generate_erlang_hyper_telegraph_map(n - 2, 2, rate_ratio=50.0,
                                                       hyper_spread=5.0, seed=_gen(18))

    raise ValueError(f"Unknown process type: {process_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TYPES = list(sm.MAP_PROCESS_TYPES)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating target MAPs (size={MAP_SIZE}) for {len(ALL_TYPES)} process types\n")

    for process_type in ALL_TYPES:
        print(f"  {process_type:<35}", end="", flush=True)
        try:
            m = generate_map(process_type)
            D0, D1 = m.D0, m.D1
            mean = sm.marginal_moment(D0, D1, 1)
            moments = compute_moments_list(D0, D1)
            payload = {"D0": D0, "D1": D1, "moments": moments}
            out = OUTPUT_DIR / f"{process_type}.pkl"
            with open(out, "wb") as f:
                pickle.dump(payload, f)
            print(f"OK  mean={mean:.4f}  {len(moments)} tuples  -> {out.name}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nDone. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
