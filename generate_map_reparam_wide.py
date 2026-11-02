"""Wide initialization samplers: same randomization as the dataset builder, at an exact size n.

generate_map_reparam.REPARAM_REGISTRY seeds the fitter with fixed generator settings
(e.g. alternating_blocks always sparse/sparse + first-phase entry, erlang_modulated always
2 modes, 50/50 block splits, MMPP from a separate direct sampler) and no diversity filter,
so it covers a narrower range of MAPs than build_map_dataset.make_candidate.

This module draws every structure exactly as make_candidate does (random block structures,
entry modes, splits, mode counts and hyperparameters) but with the total phase count fixed to n,
validates each map. As in the dataset the size varies too: each map is built with k <= n phases
(k log-uniform) and embedded exactly into n phases by state splitting (split_states; the arrival
process is unchanged). Optionally the batch is also spread the batch in the dataset's feature space
(rho1, log SCV, signed log1p|skew|, log1p kurt) by greedy farthest-point selection from an
oversampled pool -- the batch analogue of the dataset's rejection-diversity filter.

    WIDE_REGISTRY[s][0](num_maps, n, seed=None, oversample=OVERSAMPLE, size_mix=True) -> (lambda_raw, Z_P, Z_Q)
        (drop-in for REPARAM_REGISTRY)
    sample_from_feasible_structures_wide(structures, num_maps, n, seed=None, oversample=OVERSAMPLE, size_mix=True)
    make_map_n(structure, n, rng, size_mix=True) -> sampling_map.MAP      # one validated map of exactly n phases

Structures whose definition needs a factorisation n = k_modes * m_phases (erlang_modulated,
erlang_rate_spread) use k drawn from the dataset's range; when k does not divide n the chains
get m or m+1 phases (lengths differ by at most one) so the total is exactly n.
"""

from __future__ import annotations

import sys
from math import factorial
from pathlib import Path

import numpy as np

for _p in (r"C:\Users\elira\workspace\projects\map_fitting\code",
           r"C:\Users\elira\workspace\projects\one.deep.moment",
           r"C:\Users\elira\workspace\butools2\butools2\Python",
           str(Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sampling_map as sm  # noqa: E402
from generate_map_reparam import extract_map_reparam, modulated_chains  # noqa: E402

OVERSAMPLE = 3          # pool size = OVERSAMPLE * num_maps for the farthest-point spread (1 = plain sampling)
MAX_TRIES = 50          # redraws per map when a candidate is invalid / has the wrong size

# minimum total size each structure can be built with (blocks of >= 2 phases etc.)
MIN_N = {"mmpp": 2, "alternating_blocks": 4, "alternating_extreme": 4, "alternating_extreme_hyper": 4,
         "slow_switching_mmpp": 2, "erlang_modulated": 2, "hyper_telegraph": 2, "erlang_telegraph": 4,
         "erlang_rate_spread": 2, "coxian_alternating": 4, "erlang_hyper_telegraph": 5}


# =============================================================================
# Helpers
# =============================================================================

def _split(rng, n, min_a, min_b):
    """Random split n = a + b with a >= min_a, b >= min_b (as build_map_dataset._split_two, but exact)."""
    a = int(rng.integers(min_a, n - min_b + 1))
    return a, n - a


def _chain_lengths(rng, n, k_lo, k_hi):
    """k chains (k uniform in [k_lo, min(k_hi, n)]) whose lengths differ by at most one and sum to n."""
    k = int(rng.integers(k_lo, max(k_lo, min(k_hi, n)) + 1))
    base, rem = divmod(n, k)
    lengths = np.full(k, base)
    lengths[rng.permutation(k)[:rem]] += 1
    return lengths


def _erlang_modulated(rng, n):
    # dataset: k in [2, 10], m = phases per chain, conc ~ logU(1.5, 30), mode rate ~ logU(0.2, 5) * m
    lengths = _chain_lengths(rng, n, 2, 10)
    conc = float(np.exp(rng.uniform(np.log(1.5), np.log(30.0))))
    means = 1.0 / np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=len(lengths)))
    D0, D1 = modulated_chains(rng, lengths, means, conc)
    return sm.scale_map_to_mean(D0, D1, 1.0, "erlang_modulated", "erlang_modulated")


def _erlang_rate_spread(rng, n):
    # dataset: k in [2, 7], rate_spread ~ logU(50, 5000), conc ~ logU(5, 100); one rare slow mode
    lengths = _chain_lengths(rng, n, 2, 7)
    k = len(lengths)
    spread = float(np.exp(rng.uniform(np.log(50.0), np.log(5000.0))))
    conc = float(np.exp(rng.uniform(np.log(5.0), np.log(100.0))))
    slow = int(rng.integers(0, k))
    fast_base = float(np.exp(rng.uniform(np.log(1.0), np.log(5.0))))
    rates = np.array([fast_base / spread if i == slow else fast_base * float(np.exp(rng.uniform(-0.5, 0.5)))
                      for i in range(k)])
    D0, D1 = modulated_chains(rng, lengths, 1.0 / rates, conc)
    return sm.scale_map_to_mean(D0, D1, 1.0, "erlang_rate_spread", "erlang_rate_spread")


# =============================================================================
# One map of exactly n phases, randomized as in build_map_dataset.make_candidate
# =============================================================================

def _draw(structure, n, rng):
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    if structure == "mmpp":
        return sm.generate_random_mmpp(n, str(rng.choice(sm.BLOCK_STRUCTURES)), seed)
    if structure == "alternating_blocks":
        a, b = _split(rng, n, 2, 2)
        sa, sb = str(rng.choice(sm.BLOCK_STRUCTURES)), str(rng.choice(sm.BLOCK_STRUCTURES))
        return sm.generate_alternating_blocks_map(a, b, sa, sb, seed, entry_mode=str(rng.choice(sm.ALTERNATING_ENTRY_MODES)))
    if structure == "alternating_extreme":
        a, b = _split(rng, n, 2, 2)
        return sm.generate_alternating_extreme_map(a, b, seed)
    if structure == "alternating_extreme_hyper":
        a, b = _split(rng, n, 2, 2)
        return sm.generate_alternating_extreme_hyper_map(a, b, seed)
    if structure == "diverse_ph":
        return sm.generate_diverse_ph_map(str(rng.choice(sm.PH_STRUCTURES)), n, seed)
    if structure == "modified_renewal_ph":
        return sm.generate_modified_renewal_ph_map(str(rng.choice(sm.PH_STRUCTURES)), n, seed,
                                                   correlation_sign=int(rng.choice([-1, 1])))
    if structure == "random_map":
        return sm.generate_random_map_method_map(n, seed)
    if structure == "random_map_negative1":
        return sm.generate_random_map_negative1_method_map(n, seed)
    if structure == "erlang_renewal":
        return sm.generate_erlang_renewal_map(n, seed)
    if structure == "sparse_ph_renewal":
        return sm.generate_sparse_ph_renewal_map(n, seed)
    if structure == "hyper_erlang_renewal":
        return sm.generate_hyper_erlang_renewal_map(n, seed)
    if structure == "modified_hyper_erlang_renewal":
        return sm.generate_modified_hyper_erlang_renewal_map(n, seed)
    if structure == "slow_switching_mmpp":
        sw = float(np.exp(rng.uniform(np.log(0.001), np.log(0.15))))
        return sm.generate_slow_switching_mmpp(n, sw, float(rng.uniform(1.0, 3.5)), seed)
    if structure == "erlang_modulated":
        return _erlang_modulated(rng, n)
    if structure == "hyper_telegraph":
        nf, ns = _split(rng, n, 1, 1)
        lo = float(np.exp(rng.uniform(np.log(3.0), np.log(15.0))))
        hi = lo * float(np.exp(rng.uniform(np.log(2.0), np.log(10.0))))
        return sm.generate_hyper_telegraph_map(nf, ns, (lo, hi), float(rng.uniform(0.5, 2.5)), seed)
    if structure == "erlang_telegraph":
        a, b = _split(rng, n, 2, 2)
        return sm.generate_erlang_telegraph_map(a, b, float(np.exp(rng.uniform(np.log(1.2), np.log(8.0)))), seed)
    if structure == "erlang_rate_spread":
        return _erlang_rate_spread(rng, n)
    if structure == "coxian_alternating":
        a, b = _split(rng, n, 2, 2)
        return sm.generate_coxian_alternating_map(a, b, float(np.exp(rng.uniform(np.log(2.0), np.log(50.0)))),
                                                  float(np.exp(rng.uniform(np.log(1.1), np.log(5.0)))), seed)
    if structure == "erlang_hyper_telegraph":
        ke, kh = _split(rng, n, 3, 2)
        return sm.generate_erlang_hyper_telegraph_map(ke, kh, float(np.exp(rng.uniform(np.log(20.0), np.log(1000.0)))),
                                                      float(np.exp(rng.uniform(np.log(5.0), np.log(5000.0)))), seed)
    raise ValueError(f"unknown structure {structure!r}")


def map_problems(D0, D1, n, tol=1e-8):
    """List of reasons (D0, D1) is not a valid mean-1 MAP of size n; empty if valid."""
    out = []
    if D0.shape != (n, n) or D1.shape != (n, n):
        return [f"size {D0.shape[0]} != {n}"]
    if not (np.all(np.isfinite(D0)) and np.all(np.isfinite(D1))):
        return ["non-finite"]
    off = D0 - np.diag(np.diag(D0))
    if off.min() < -tol or D1.min() < -tol:
        out.append("negative rate")
    if np.diag(D0).max() >= 0:
        out.append("non-negative diagonal")
    scale = np.abs(np.diag(D0)).max()
    if np.abs((D0 + D1).sum(1)).max() > 1e-8 * scale:
        out.append("rows of D0+D1 do not sum to 0")
    if D1.sum() <= 0:
        out.append("no arrivals")
    if not out:
        try:
            m1 = float(stats(D0, D1)["m1"])
            if abs(m1 - 1) > 1e-6:
                out.append(f"mean {m1:.4g} != 1")
        except Exception as exc:
            out.append(f"singular ({exc})")
    return out


def split_states(D0, D1, n, rng, keep_self=False):
    """Exact embedding of a k-state MAP into n >= k states by duplicating states (ordinary lumpability).

    A copy i' of state i gets i's outgoing rates; every transition INTO i is split between i and i'
    with one fraction f (the same for all source rows, so a rank-1 D1 stays rank 1). Lumping each
    copy back onto its original gives the original MAP, so the arrival process -- all marginal and
    joint moments -- is unchanged. keep_self=True sends i->i arrivals of a copy to the copy itself
    (keeps an MMPP's D1 diagonal).
    """
    D0, D1 = D0.copy(), D1.copy()
    while len(D0) < n:
        k = len(D0); i = int(rng.integers(k)); f = float(rng.uniform(0.2, 0.8))
        A0, A1 = np.zeros((k + 1, k + 1)), np.zeros((k + 1, k + 1))
        A0[:k, :k], A1[:k, :k] = D0, D1
        A0[k, :k], A1[k, :k] = D0[i], D1[i]                 # copy i's outgoing rates
        A0[k, k], A0[k, i] = D0[i, i], 0.0                  # copy's own diagonal; no copy -> i jump
        for A in (A0, A1):                                  # split transitions into i between i and the copy
            inflow = A[:, i].copy()
            if A is A0:
                inflow[[i, k]] = 0.0                        # diagonals are not transitions
            A[:, i] -= (1 - f) * inflow
            A[:, k] += (1 - f) * inflow
        if keep_self:                                       # copy's self-arrival goes to the copy only
            A1[k, k] += A1[k, i]; A1[k, i] = 0.0
            A1[i, i] += A1[i, k]; A1[i, k] = 0.0
        D0, D1 = A0, A1
    return D0, D1


def _min_size(structure):
    return max(MIN_N.get(structure, 1), 2)


def make_map_n(structure, n, rng, size_mix=True):
    """One validated MAP of exactly n phases for the given structure.

    size_mix=True: as in the dataset, the structure is built with k phases, k log-uniform in
    [min size, n], and embedded exactly into n phases with split_states, so the batch covers the
    shapes of small maps (e.g. Erlang-k renewal with SCV 1/k) as well as large ones.
    """
    if n < MIN_N.get(structure, 1):
        raise ValueError(f"{structure} needs n >= {MIN_N[structure]}")
    last = None
    for _ in range(MAX_TRIES):
        k = n
        if size_mix and n > _min_size(structure):
            k = int(np.clip(round(np.exp(rng.uniform(np.log(_min_size(structure)), np.log(n)))), _min_size(structure), n))
        try:
            m = _draw(structure, k, rng)
        except Exception as exc:          # generator rejection (e.g. random_map_negative1 out of tries)
            last = str(exc)
            continue
        probs = map_problems(m.D0, m.D1, k)
        if probs:
            last = "; ".join(probs)
            continue
        if k < n:
            D0, D1 = split_states(m.D0, m.D1, n, rng, keep_self=structure in ("mmpp", "slow_switching_mmpp"))
            m = sm.MAP(D0, D1, m.process_type, m.structure)
            probs = map_problems(m.D0, m.D1, n)
            if probs:
                last = "embedding: " + "; ".join(probs)
                continue
        return m
    raise RuntimeError(f"{structure} n={n}: no valid map after {MAX_TRIES} draws ({last})")


# =============================================================================
# Features and diversity spread
# =============================================================================

def stats(D0, D1):
    """m1, scv, skew, kurt, rho1..rho3 of a MAP (arrival-epoch stationary)."""
    n = len(D0); M = np.linalg.inv(-D0); P = M @ D1; o = np.ones(n)
    Q = (D0 + D1).T.copy(); Q[-1, :] = 1.0; b = np.zeros(n); b[-1] = 1.0
    pi = np.linalg.solve(Q, b); beta = pi @ D1; beta = beta / beta.sum()
    mu, Mk = [1.0], np.eye(n)
    for k in range(1, 5):
        Mk = Mk @ M; mu.append(factorial(k) * beta @ Mk @ o)
    m1, m2, m3, m4 = mu[1:]
    var = m2 - m1 ** 2; sd = np.sqrt(max(var, 1e-300))
    rhos, Pk = [], np.eye(n)
    for _ in range(3):
        Pk = Pk @ P; rhos.append((beta @ M @ Pk @ M @ o - m1 ** 2) / var)
    return dict(m1=m1, scv=var / m1 ** 2, skew=(m3 - 3 * m2 * m1 + 2 * m1 ** 3) / sd ** 3,
                kurt=(m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4) / var ** 2 - 3,
                rho1=rhos[0], rho2=rhos[1], rho3=rhos[2])


def feature_vec(D0, D1):
    s = stats(D0, D1)
    return np.array([s["rho1"], np.log(max(s["scv"], 1e-6)), np.sign(s["skew"]) * np.log1p(abs(s["skew"])),
                     np.log1p(max(s["kurt"], 0.0))])


def farthest_point(F, k, rng):
    """Indices of k rows of F chosen greedily to maximise the minimum pairwise distance."""
    F = np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=-50.0)
    chosen = [int(rng.integers(len(F)))]
    d = np.linalg.norm(F - F[chosen[0]], axis=1)
    for _ in range(k - 1):
        j = int(d.argmax()); chosen.append(j)
        d = np.minimum(d, np.linalg.norm(F - F[j], axis=1))
    return np.array(chosen)


# =============================================================================
# Batch samplers
# =============================================================================

def sample_wide(structure, num_maps, n, seed=None, oversample=OVERSAMPLE, size_mix=True):
    """(lambda_raw, Z_P, Z_Q) for num_maps validated maps of exactly n phases."""
    rng = np.random.default_rng(seed)
    pool = [make_map_n(structure, n, rng, size_mix) for _ in range(max(num_maps, int(round(oversample * num_maps))))]
    if len(pool) > num_maps:
        F = np.array([feature_vec(m.D0, m.D1) for m in pool])
        pool = [pool[i] for i in farthest_point(F, num_maps, rng)]
    parts = [extract_map_reparam(m.D0, m.D1) for m in pool]
    return tuple(np.stack([p[i] for p in parts]) for i in range(3))


def _make_sampler(structure):
    def sampler(num_maps, n, seed=None, oversample=OVERSAMPLE, size_mix=True):
        return sample_wide(structure, num_maps, n, seed, oversample, size_mix)
    sampler.__name__ = f"sample_{structure}_wide"
    return sampler


WIDE_REGISTRY = {s: (_make_sampler(s), s) for s in sm.MAP_PROCESS_TYPES}


def sample_from_feasible_structures_wide(structures, num_maps, n, seed=None, oversample=OVERSAMPLE, size_mix=True):
    """Same split as generate_map_reparam.sample_from_feasible_structures, with the wide samplers."""
    if not structures:
        raise ValueError("structures list is empty")
    rng = np.random.default_rng(seed)
    base, rem = divmod(num_maps, len(structures))
    parts = []
    for i, s in enumerate(structures):
        count = base + (1 if i < rem else 0)
        if count:
            parts.append(sample_wide(s, count, n, int(rng.integers(0, 2 ** 31)), oversample, size_mix))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
