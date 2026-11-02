"""Batch reparametrisation samplers for all 19 MAP process types.

Universal representation: every MAP (D0, D1) is encoded as
    (lambda_raw, Z_P, Z_Q)
where
    rates    = lambda_raw ** 2                        # total leaving rate per state
    P        = softmax_rows(Z_P)                      # arrival fraction (diag) + no-arrival routing (off-diag)
    Q        = softmax_rows(Z_Q)                      # arrival restart distribution
    D0[i,j]  = rates[i] * P[i,j]   (j != i)
    D0[i,i]  = -rates[i]
    D1[i,:]  = rates[i] * P[i,i] * Q[i,:]

Structural zeros (forbidden transitions) are encoded as -SOFTMAX_LARGE in Z_P / Z_Q
so that softmax gives ~0 and gradient steps cannot easily escape the constraint.

Each public function has signature:
    sample_<type>_reparam_tensors(num_maps, n, seed=None)
        -> (lambda_raw, Z_P, Z_Q)  shapes (num_maps,n), (num_maps,n,n), (num_maps,n,n)

For MMPP the tensors are sampled directly in reparametrisation space (matching the
original generator's marginal distributions).  For all other types the original
sampling_map generators are called and the reparametrisation parameters are extracted
analytically from the resulting (D0, D1).
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

SOFTMAX_LARGE = 500.0
_EPS = 1e-14


# =============================================================================
# Core utilities
# =============================================================================

def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    Z_s = Z - Z.max(axis=-1, keepdims=True)
    e = np.exp(Z_s)
    return e / e.sum(axis=-1, keepdims=True)


def extract_map_reparam(
    D0: np.ndarray,
    D1: np.ndarray,
    large: float = SOFTMAX_LARGE,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytically convert (D0, D1) -> (lambda_raw, Z_P, Z_Q).

    The conversion is exact: softmax(Z_P) = P and softmax(Z_Q) = Q, so
    reconstructing (D0, D1) from the returned tensors recovers the originals
    to floating-point precision.
    """
    n = D0.shape[0]
    rates = -np.diag(D0)                                # lambda_i^2
    lambda_raw = np.sqrt(np.maximum(rates, 1e-10))

    # P matrix: off-diagonal from D0, diagonal = arrival fraction
    P = D0.copy()
    np.fill_diagonal(P, 0.0)
    P /= np.maximum(rates[:, None], eps)                # P[i,j] = D0[i,j]/rates[i]
    np.fill_diagonal(P, D1.sum(axis=1) / np.maximum(rates, eps))

    # Z_P = log(P), clamped to -large where P ~ 0
    Z_P = np.where(P > eps, np.log(np.maximum(P, eps)), -large)

    # Q matrix: row-normalised D1
    D1_sums = D1.sum(axis=1, keepdims=True)
    valid = (D1_sums > eps).squeeze()
    Q = np.zeros_like(D1)
    Q[valid] = D1[valid] / D1_sums[valid]
    Q[~valid] = 1.0 / n                                 # no arrivals: flat (won't be used)

    # Z_Q = log(Q), clamped to -large where Q ~ 0
    Z_Q = np.where(Q > eps, np.log(np.maximum(Q, eps)), -large)

    return lambda_raw, Z_P, Z_Q


def _stack_reparam(maps: list, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and stack (lambda_raw, Z_P, Z_Q) for a list of MAP objects.

    If n is given, every map must have exactly n phases (a generator returning another size,
    e.g. the random_map helper's retry at a random size, is reported here instead of failing
    later as a shape mismatch in the fitter).
    """
    if n is not None:
        for m in maps:
            if m.D0.shape[0] != n:
                raise ValueError(f"{m.process_type}: sampler returned {m.D0.shape[0]} states, expected {n}")
    parts = [extract_map_reparam(m.D0, m.D1) for m in maps]
    return (
        np.stack([p[0] for p in parts]),
        np.stack([p[1] for p in parts]),
        np.stack([p[2] for p in parts]),
    )


# =============================================================================
# Per-type tensor samplers
# =============================================================================

# ------------------------------------------------------------------
# 1. mmpp  (direct reparametrisation sampler, matches original marginals)
# ------------------------------------------------------------------

def sample_mmpp_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MMPP: sample (lambda_raw, Z_P, Z_Q) in reparametrisation space directly."""
    from generate_mmpp_reparam import sample_mmpp_reparam_tensors as _impl
    return _impl(num_maps, n, seed)


# ------------------------------------------------------------------
# 2. alternating_blocks
# ------------------------------------------------------------------

def sample_alternating_blocks_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    na, nb = n // 2, n - n // 2
    maps = [
        sm.generate_alternating_blocks_map(
            na, nb, "sparse", "sparse",
            seed=int(rng.integers(0, 2**31)),
            entry_mode="first_phase",
        )
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 3. alternating_extreme
# ------------------------------------------------------------------

def sample_alternating_extreme_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    na, nb = n // 2, n - n // 2
    maps = [
        sm.generate_alternating_extreme_map(na, nb, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 4. alternating_extreme_hyper
# ------------------------------------------------------------------

def sample_alternating_extreme_hyper_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    na, nb = n // 2, n - n // 2
    maps = [
        sm.generate_alternating_extreme_hyper_map(na, nb, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 5. diverse_ph  (uses external PH sampler)
# ------------------------------------------------------------------

def sample_diverse_ph_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ph_structures = ["coxian_ph", "hyper_erlang_ph", "general_ph", "modified_hyper_erlang_ph"]
    maps = [
        sm.generate_diverse_ph_map(
            ph_structures[i % len(ph_structures)], n,
            seed=int(rng.integers(0, 2**31)),
        )
        for i in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 6. modified_renewal_ph  (uses external PH sampler)
# ------------------------------------------------------------------

def sample_modified_renewal_ph_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ph_structures = ["coxian_ph", "hyper_erlang_ph", "general_ph", "modified_hyper_erlang_ph"]
    maps = [
        sm.generate_modified_renewal_ph_map(
            ph_structures[i % len(ph_structures)], n,
            seed=int(rng.integers(0, 2**31)),
            correlation_sign=1 if i % 2 == 0 else -1,
        )
        for i in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 7. random_map
# ------------------------------------------------------------------

def sample_random_map_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_random_map_method_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 8. random_map_negative1
# ------------------------------------------------------------------

def sample_random_map_negative1_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_random_map_negative1_method_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 9. erlang_renewal
# ------------------------------------------------------------------

def sample_erlang_renewal_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_erlang_renewal_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 10. sparse_ph_renewal  (uses external PH sampler)
# ------------------------------------------------------------------

def sample_sparse_ph_renewal_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_sparse_ph_renewal_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 11. hyper_erlang_renewal  (uses external PH sampler)
# ------------------------------------------------------------------

def sample_hyper_erlang_renewal_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_hyper_erlang_renewal_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 12. modified_hyper_erlang_renewal  (uses external PH sampler)
# ------------------------------------------------------------------

def sample_modified_hyper_erlang_renewal_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = [
        sm.generate_modified_hyper_erlang_renewal_map(n, seed=int(rng.integers(0, 2**31)))
        for _ in range(num_maps)
    ]
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 13. slow_switching_mmpp
# ------------------------------------------------------------------

def sample_slow_switching_mmpp_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = []
    for _ in range(num_maps):
        switching_scale = float(np.exp(rng.uniform(np.log(0.001), np.log(0.15))))
        rate_spread = float(rng.uniform(1.0, 3.5))
        maps.append(
            sm.generate_slow_switching_mmpp(
                n, switching_scale=switching_scale, rate_spread=rate_spread,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 14. erlang_modulated  (k=2 modes, m=n//2 phases each → total n)
# ------------------------------------------------------------------

def modulated_chains(
    rng: np.random.Generator, lengths: np.ndarray, mode_means: np.ndarray, concentration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Erlang chains of the given lengths (they may differ, so any total n is possible).

    Chain i has mean mode_means[i]; an arrival at its last phase restarts chain j with
    Dirichlet probabilities concentrated (weight `concentration`) on j = i.  Same construction
    as sm.generate_erlang_modulated_map / generate_erlang_rate_spread_map, which only allow
    equal lengths (total k * m).
    """
    k, n = len(lengths), int(np.sum(lengths))
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    D0, D1 = np.zeros((n, n)), np.zeros((n, n))
    for mode in range(k):
        s, m = starts[mode], int(lengths[mode])
        rate = m / mode_means[mode]
        for ph in range(m):
            D0[s + ph, s + ph] = -rate
            if ph < m - 1:
                D0[s + ph, s + ph + 1] = rate
        conc = np.ones(k)
        conc[mode] = concentration
        D1[s + m - 1, starts] = rate * rng.dirichlet(conc)
    return D0, D1


def _two_chain_lengths(n: int) -> np.ndarray:
    """Two chains with n // 2 and n - n // 2 phases (total exactly n; n = 1 gives one chain)."""
    return np.array([n // 2, n - n // 2]) if n >= 2 else np.array([1])


def sample_erlang_modulated_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same distribution as sm.generate_erlang_modulated_map(2, n // 2, conc): mode rate
    # ~ logU(0.2, 5) per phase-normalised chain, i.e. chain mean = 1 / rate draw. Built
    # directly so that odd n gives chains of n // 2 and n // 2 + 1 phases (total n, not n - 1).
    rng = np.random.default_rng(seed)
    lengths = _two_chain_lengths(n)
    maps = []
    for _ in range(num_maps):
        conc = float(np.exp(rng.uniform(np.log(1.5), np.log(30.0))))
        means = 1.0 / np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=len(lengths)))
        D0, D1 = modulated_chains(rng, lengths, means, conc)
        maps.append(sm.scale_map_to_mean(D0, D1, 1.0, "erlang_modulated", "erlang_modulated"))
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 15. hyper_telegraph
# ------------------------------------------------------------------

def sample_hyper_telegraph_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nf, ns = n // 2, n - n // 2
    maps = []
    for _ in range(num_maps):
        rr_lo = float(np.exp(rng.uniform(np.log(3.0), np.log(15.0))))
        rr_hi = rr_lo * float(np.exp(rng.uniform(np.log(2.0), np.log(10.0))))
        spread = float(rng.uniform(0.5, 2.5))
        maps.append(
            sm.generate_hyper_telegraph_map(
                nf, ns, rate_ratio_range=(rr_lo, rr_hi),
                within_block_spread=spread,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 16. erlang_telegraph
# ------------------------------------------------------------------

def sample_erlang_telegraph_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ka, kb = max(1, n // 2), max(1, n - n // 2)
    maps = []
    for _ in range(num_maps):
        rr = float(np.exp(rng.uniform(np.log(1.2), np.log(8.0))))
        maps.append(
            sm.generate_erlang_telegraph_map(
                ka, kb, rate_ratio=rr,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 17. erlang_rate_spread  (k=2, m=n//2 → total n)
# ------------------------------------------------------------------

def sample_erlang_rate_spread_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same distribution as sm.generate_erlang_rate_spread_map(2, n // 2, rs, conc): one rare
    # slow mode at fast_base / rate_spread, the other at fast_base * e^U(-0.5, 0.5). Built
    # directly so that odd n gives chains of n // 2 and n // 2 + 1 phases (total n, not n - 1).
    rng = np.random.default_rng(seed)
    lengths = _two_chain_lengths(n)
    k = len(lengths)
    maps = []
    for _ in range(num_maps):
        rs = float(np.exp(rng.uniform(np.log(50.0), np.log(5000.0))))
        conc = float(np.exp(rng.uniform(np.log(5.0), np.log(100.0))))
        slow = int(rng.integers(0, k))
        fast_base = float(np.exp(rng.uniform(np.log(1.0), np.log(5.0))))
        rates = np.array([fast_base / rs if i == slow else fast_base * float(np.exp(rng.uniform(-0.5, 0.5)))
                          for i in range(k)])
        D0, D1 = modulated_chains(rng, lengths, 1.0 / rates, conc)
        maps.append(sm.scale_map_to_mean(D0, D1, 1.0, "erlang_rate_spread", "erlang_rate_spread"))
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 18. coxian_alternating
# ------------------------------------------------------------------

def sample_coxian_alternating_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ka, kb = max(2, n // 2), max(2, n - n // 2)
    maps = []
    for _ in range(num_maps):
        rr = float(np.exp(rng.uniform(np.log(2.0), np.log(50.0))))
        es = float(np.exp(rng.uniform(np.log(1.1), np.log(5.0))))
        maps.append(
            sm.generate_coxian_alternating_map(
                ka, kb, rate_ratio=rr, exit_shape=es,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return _stack_reparam(maps, n)


# ------------------------------------------------------------------
# 19. erlang_hyper_telegraph  (k_erlang=n-2, k_hyper=2 → total n)
# ------------------------------------------------------------------

def sample_erlang_hyper_telegraph_reparam_tensors(
    num_maps: int, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k_erlang = max(1, n - 2)
    k_hyper  = 2
    maps = []
    for _ in range(num_maps):
        rr = float(np.exp(rng.uniform(np.log(20.0), np.log(1000.0))))
        hs = float(np.exp(rng.uniform(np.log(5.0), np.log(5000.0))))
        maps.append(
            sm.generate_erlang_hyper_telegraph_map(
                k_erlang, k_hyper, rate_ratio=rr, hyper_spread=hs,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return _stack_reparam(maps, n)


# =============================================================================
# Registry: type name -> (sampler function, process_type string)
# =============================================================================

REPARAM_REGISTRY: dict[str, tuple] = {
    "mmpp":                         (sample_mmpp_reparam_tensors,                    "mmpp"),
    "alternating_blocks":           (sample_alternating_blocks_reparam_tensors,      "alternating_blocks"),
    "alternating_extreme":          (sample_alternating_extreme_reparam_tensors,     "alternating_extreme"),
    "alternating_extreme_hyper":    (sample_alternating_extreme_hyper_reparam_tensors, "alternating_extreme_hyper"),
    "diverse_ph":                   (sample_diverse_ph_reparam_tensors,              "diverse_ph"),
    "modified_renewal_ph":          (sample_modified_renewal_ph_reparam_tensors,     "modified_renewal_ph"),
    "random_map":                   (sample_random_map_reparam_tensors,              "random_map"),
    "random_map_negative1":         (sample_random_map_negative1_reparam_tensors,    "random_map_negative1"),
    "erlang_renewal":               (sample_erlang_renewal_reparam_tensors,          "erlang_renewal"),
    "sparse_ph_renewal":            (sample_sparse_ph_renewal_reparam_tensors,       "sparse_ph_renewal"),
    "hyper_erlang_renewal":         (sample_hyper_erlang_renewal_reparam_tensors,    "hyper_erlang_renewal"),
    "modified_hyper_erlang_renewal":(sample_modified_hyper_erlang_renewal_reparam_tensors, "modified_hyper_erlang_renewal"),
    "slow_switching_mmpp":          (sample_slow_switching_mmpp_reparam_tensors,     "slow_switching_mmpp"),
    "erlang_modulated":             (sample_erlang_modulated_reparam_tensors,        "erlang_modulated"),
    "hyper_telegraph":              (sample_hyper_telegraph_reparam_tensors,         "hyper_telegraph"),
    "erlang_telegraph":             (sample_erlang_telegraph_reparam_tensors,        "erlang_telegraph"),
    "erlang_rate_spread":           (sample_erlang_rate_spread_reparam_tensors,      "erlang_rate_spread"),
    "coxian_alternating":           (sample_coxian_alternating_reparam_tensors,      "coxian_alternating"),
    "erlang_hyper_telegraph":       (sample_erlang_hyper_telegraph_reparam_tensors,  "erlang_hyper_telegraph"),
}


# =============================================================================
# Multi-structure sampler
# =============================================================================

def sample_from_feasible_structures(
    structures: list[str],
    num_maps: int,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample reparametrisation tensors from a list of feasible MAP structures.

    num_maps is split as evenly as possible across structures; any remainder is
    distributed one extra to the first structures (e.g. 8 maps, 3 structures
    -> [3, 3, 2]).

    Parameters
    ----------
    structures : list of process-type name strings (e.g. from feasible_structures())
    num_maps   : total number of MAPs to generate
    n          : MAP state-space size
    seed       : optional RNG seed for reproducibility

    Returns
    -------
    lams : ndarray (num_maps, n)        -- lambda_raw
    ps   : ndarray (num_maps, n, n)     -- Z_P
    qs   : ndarray (num_maps, n, n)     -- Z_Q
    """
    if not structures:
        raise ValueError("structures list is empty")

    rng = np.random.default_rng(seed)

    base, remainder = divmod(num_maps, len(structures))
    counts = [base + (1 if i < remainder else 0) for i in range(len(structures))]

    lams_list, ps_list, qs_list = [], [], []
    for structure, count in zip(structures, counts):
        if count == 0:
            continue
        sampler, _ = REPARAM_REGISTRY[structure]
        s = int(rng.integers(0, 2**31))
        lam, zp, zq = sampler(count, n, seed=s)
        lams_list.append(lam)
        ps_list.append(zp)
        qs_list.append(zq)

    return (
        np.concatenate(lams_list, axis=0),
        np.concatenate(ps_list,   axis=0),
        np.concatenate(qs_list,   axis=0),
    )


# =============================================================================
# Example generation: 5 maps of size 10 per type → data/map_examples/
# =============================================================================

def save_map_examples(
    out_dir: str | Path = "data/map_examples",
    n: int = 10,
    num_maps: int = 5,
    seed: int = 0,
) -> None:
    """Generate num_maps examples of size n for every MAP type and pickle them.

    Each pickle contains a list of dicts, one per map:
        {
            'process_type': str,
            'lambda_raw'  : ndarray (n,),
            'Z_P'         : ndarray (n, n),
            'Z_Q'         : ndarray (n, n),
        }
    """
    out_path = Path(__file__).parent / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for type_name, (sampler, process_type) in REPARAM_REGISTRY.items():
        type_seed = int(rng.integers(0, 2**31))
        print(f"  {type_name:<35}", end="", flush=True)
        try:
            lambda_raws, Z_Ps, Z_Qs = sampler(num_maps, n, seed=type_seed)
            records = [
                {
                    "process_type": process_type,
                    "lambda_raw":   lambda_raws[k],
                    "Z_P":          Z_Ps[k],
                    "Z_Q":          Z_Qs[k],
                }
                for k in range(num_maps)
            ]
            pkl_path = out_path / f"{type_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(records, f)
            print(f"OK  -> {pkl_path.name}  (lambda_raw {lambda_raws.shape}, Z_P {Z_Ps.shape})")
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    # print(f"Generating 5 examples of size 10 for each of {len(REPARAM_REGISTRY)} MAP types...\n")
    # save_map_examples()
    # print("\nDone.")
    pass
