"""MAP structure feasibility selector.

Given a target (rho1, scv, skewness, kurtosis), returns which of the 19 MAP
process types are capable of producing a MAP with statistics close to those values.

Usage
-----
    from map_structure_selector import build_coverage, feasible_structures

    # One-time: build and save coverage dataset (~1-2 min)
    build_coverage(save_path="data/coverage.pkl")

    # Query
    candidates = feasible_structures(rho1=0.3, scv=4.0, skewness=8.0, kurtosis=120.0)
    print(candidates)

Method
------
1. Hard analytical rules immediately exclude structures that are provably infeasible
   (e.g. rho1 < 0 → MMPP excluded; SCV >= 1 required for MMPP).
2. For the remaining structures, a KNN check in normalised log-space determines
   whether the query point falls within the structure's empirical coverage region.
   A structure is accepted if its nearest sample is within `radius` standard
   deviations of the within-structure nearest-neighbour distances.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

DEFAULT_COVERAGE_PATH = Path(__file__).parent / "data" / "coverage.pkl"

# ---------------------------------------------------------------------------
# Statistics from target_map
# ---------------------------------------------------------------------------

def stats_from_target_map(target_map: dict) -> tuple[float, float, float, float]:
    """Compute (rho1, scv, skewness, kurtosis) from a target_map dict.

    Uses the 'moments' list when present (preferred — no extra computation).
    Falls back to computing directly from D0/D1 if moments are absent or
    lack 4th-order marginals.

    Returns
    -------
    rho1     : lag-1 autocorrelation  Cov(X_n, X_{n+1}) / Var[X]
    scv      : squared coefficient of variation  Var[X] / E[X]^2
    skewness : standardised 3rd central moment
    kurtosis : excess kurtosis  (4th central moment / Var^2) - 3
    """
    moments = target_map.get("moments", [])
    mom = {(i, j, k): v for i, j, k, v, *_ in moments}

    have_marginals = all((m, 0, 0) in mom for m in range(1, 5))
    have_joint     = (1, 1, 1) in mom

    if have_marginals and have_joint:
        mu1     = mom[(1, 0, 0)]
        mu2     = mom[(2, 0, 0)]
        mu3     = mom[(3, 0, 0)]
        mu4     = mom[(4, 0, 0)]
        joint11 = mom[(1, 1, 1)]
    else:
        from math import factorial
        D0, D1 = target_map["D0"], target_map["D1"]
        raw  = sm._raw_moments(D0, D1, max_power=4)
        mu1, mu2, mu3, mu4 = raw[1], raw[2], raw[3], raw[4]

        minus_D0 = -D0
        beta = sm.arrival_epoch_distribution(D0, D1)
        ones = np.ones(D0.shape[0])
        v = np.linalg.solve(minus_D0, ones)
        v = D1 @ v
        v = np.linalg.solve(minus_D0, np.linalg.solve(minus_D0, v))
        joint11 = float(beta @ v)

    var = mu2 - mu1 ** 2
    std = float(np.sqrt(max(var, 0.0)))

    scv      = var / mu1 ** 2
    cm3      = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
    skewness = cm3 / std ** 3 if std > 0 else 0.0
    cm4      = mu4 - 4 * mu3 * mu1 + 6 * mu2 * mu1 ** 2 - 3 * mu1 ** 4
    kurtosis = cm4 / var ** 2 - 3 if var > 0 else 0.0
    rho1     = (joint11 - mu1 ** 2) / var if var > 0 else 0.0

    return float(rho1), float(scv), float(skewness), float(kurtosis)


# ---------------------------------------------------------------------------
# Hard analytical exclusion rules
# ---------------------------------------------------------------------------

def _hard_exclude(process_type: str, rho1: float, scv: float,
                  skewness: float, kurtosis: float) -> bool:
    """Return True if process_type is provably UNABLE to produce these stats."""

    # --- rho1 sign constraints ---
    # Strictly always-negative autocorrelation structures
    always_negative = {
        "alternating_extreme", "erlang_telegraph",
        "hyper_telegraph", "coxian_alternating", "erlang_hyper_telegraph",
    }
    if process_type in always_negative and rho1 > 0.05:
        return True

    # Strictly always-positive or zero autocorrelation structures
    always_nonneg = {
        "mmpp", "slow_switching_mmpp", "erlang_modulated",
        "erlang_renewal", "erlang_rate_spread",
        "diverse_ph", "sparse_ph_renewal",
        "hyper_erlang_renewal", "modified_hyper_erlang_renewal",
    }
    if process_type in always_nonneg and rho1 < -0.05:
        return True

    # Renewal processes: rho1 must be ~0
    renewal = {
        "erlang_renewal", "diverse_ph", "sparse_ph_renewal",
        "hyper_erlang_renewal", "modified_hyper_erlang_renewal",
    }
    if process_type in renewal and abs(rho1) > 0.10:
        return True

    # --- SCV constraints ---
    # MMPP always has SCV >= 1 (mixture of exponentials with different rates)
    if process_type in {"mmpp", "slow_switching_mmpp"} and scv < 0.90:
        return True

    # Erlang renewal: SCV = 1/k, so SCV > 1 is impossible
    if process_type == "erlang_renewal" and scv > 1.05:
        return True

    # Erlang-based chains keep SCV well below 1 when the rate ratio is moderate
    if process_type == "erlang_telegraph" and scv > 1.10:
        return True

    if process_type == "erlang_modulated" and scv > 1.05:
        return True

    # --- Slow switching: rho1 must be high ---
    if process_type == "slow_switching_mmpp" and rho1 < 0.20:
        return True

    return False


# ---------------------------------------------------------------------------
# Feature transform: map raw stats to a normalised space for distance queries
# ---------------------------------------------------------------------------

def _transform(rho1: float, scv: float, skewness: float, kurtosis: float
               ) -> np.ndarray:
    """Transform (rho1, scv, skewness, kurtosis) to a normalised 4D feature vector.

    rho1      : kept linear (already in [-1, 1])
    scv       : log (heavy right tail)
    skewness  : sign * log(1 + |skewness|)
    kurtosis  : log(1 + kurtosis)
    """
    return np.array([
        rho1,
        np.log(max(scv, 1e-6)),
        np.sign(skewness) * np.log1p(abs(skewness)),
        np.log1p(max(kurtosis, 0.0)),
    ])


# ---------------------------------------------------------------------------
# Build coverage dataset
# ---------------------------------------------------------------------------

def build_coverage(
    num_per_type: int = 100_000,
    min_size: int = 5,
    max_size: int = 50,
    seed: int = 0,
    save_path: str | Path = DEFAULT_COVERAGE_PATH,
    batch_size: int = 2_000,
) -> dict:
    """Sample MAPs from all 19 types, compute statistics, save to pickle.

    Uses batched generation so memory stays bounded regardless of num_per_type.
    With 100k samples per type the resulting KD-trees give tight coverage
    estimates and fast (microsecond) query times.

    Returns a dict:  process_type -> ndarray of shape (N, 4) in transformed space.
    """
    from scipy.spatial import KDTree

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, np.ndarray] = {}
    trees:    dict[str, KDTree]     = {}

    for i, process_type in enumerate(sm.MAP_PROCESS_TYPES):
        print(f"  [{i+1:2d}/19] {process_type:<35}", end="", flush=True)
        try:
            all_rows = []
            remaining = num_per_type
            batch_seed = seed + i * 1009
            while remaining > 0:
                n = min(batch_size, remaining)
                maps = sm.generate_experiment_sample(
                    process_type=process_type,
                    num_maps=n,
                    min_size=min_size,
                    max_size=max_size,
                    seed=batch_seed,
                )
                all_rows.extend(sm.compute_feature_rows(maps))
                remaining -= n
                batch_seed += 1

            if not all_rows:
                print("no valid rows")
                continue

            feats = np.array([
                _transform(r.ac1, r.scv, r.skewness, r.kurtosis)
                for r in all_rows
            ])
            coverage[process_type] = feats
            print(f"{len(all_rows):6d} samples  "
                  f"rho1=[{min(r.ac1 for r in all_rows):.2f},{max(r.ac1 for r in all_rows):.2f}]  "
                  f"scv=[{min(r.scv for r in all_rows):.1f},{max(r.scv for r in all_rows):.1f}]")
        except Exception as exc:
            print(f"FAILED: {exc}")

    # Pre-build KD-trees in normalised space
    all_feats = np.concatenate(list(coverage.values()), axis=0)
    scale = all_feats.std(axis=0)
    scale[scale < 1e-8] = 1.0

    for process_type, feats in coverage.items():
        trees[process_type] = KDTree(feats / scale)

    payload = {"coverage": coverage, "trees": trees, "scale": scale}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved coverage dataset ({num_per_type} per type) -> {save_path}")
    return payload


def load_coverage(path: str | Path = DEFAULT_COVERAGE_PATH) -> dict:
    print(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------------

def feasible_structures(
    rho1: float,
    scv: float,
    skewness: float,
    kurtosis: float,
    coverage: dict | None = None,
    coverage_path: str | Path = DEFAULT_COVERAGE_PATH,
    radius: float = 2.0,
) -> list[str]:
    """Return MAP process types that can plausibly generate the given statistics.

    Parameters
    ----------
    rho1, scv, skewness, kurtosis : float
        Target statistics.
    coverage : dict, optional
        Pre-loaded payload from build_coverage / load_coverage.
        If None, loaded from coverage_path.
    radius : float
        Acceptance threshold in units of the within-structure median NN distance.
        With 100k samples this can be tightened to ~1.0; default 2.0 is generous.

    Returns
    -------
    List of process type strings, sorted by distance to nearest sample (closest first).
    """
    if coverage is None:
        coverage = load_coverage(coverage_path)

    # Support both old format (plain dict of arrays) and new format (with trees/scale)
    if "trees" in coverage:
        trees  = coverage["trees"]
        scale  = coverage["scale"]
        feat_dict = coverage["coverage"]
    else:
        # Legacy: rebuild trees on the fly
        from scipy.spatial import KDTree
        feat_dict = coverage
        all_feats = np.concatenate(list(feat_dict.values()), axis=0)
        scale = all_feats.std(axis=0)
        scale[scale < 1e-8] = 1.0
        trees = {pt: KDTree(f / scale) for pt, f in feat_dict.items()}

    query_norm = _transform(rho1, scv, skewness, kurtosis) / scale

    results = []
    for process_type, tree in trees.items():
        # 1. Hard exclusion
        if _hard_exclude(process_type, rho1, scv, skewness, kurtosis):
            continue

        # 2. KD-tree nearest-neighbour query
        min_dist, _ = tree.query(query_norm, k=1)

        # Threshold: radius × median NN distance within the structure
        # (structure-specific density measure — tighter where samples are dense)
        sample_nn_dists, _ = tree.query(
            feat_dict[process_type] / scale,
            k=2,  # k=2: first match is self, second is true NN
        )
        median_nn = float(np.median(sample_nn_dists[:, 1]))
        threshold = radius * median_nn

        if min_dist <= threshold:
            results.append((process_type, float(min_dist)))

    results.sort(key=lambda x: x[1])
    return [pt for pt, _ in results]


def feasible_structures_crude(
    rho1: float,
    scv: float,
    skewness: float,
    kurtosis: float,
    coverage: dict | None = None,
    coverage_path: str | Path = DEFAULT_COVERAGE_PATH,
    rho1_tol: float = 0.15,
    scv_factor: float = 2.0,
) -> list[str]:
    """Robust fallback: accept a structure if rho1 and scv fall within its
    observed range (with generous tolerance), after hard exclusion rules.

    Much less sensitive to coverage sparsity than the KNN method.

    Parameters
    ----------
    rho1_tol   : additive tolerance on the rho1 range edges
    scv_factor : multiplicative buffer on the scv range edges (e.g. 2.0 means
                 accept if scv is within [min/2, max*2] of observed range)
    """
    if coverage is None:
        coverage = load_coverage(coverage_path)

    feat_dict = coverage.get("coverage", coverage)

    results = []
    for process_type, feats in feat_dict.items():
        if _hard_exclude(process_type, rho1, scv, skewness, kurtosis):
            continue

        # feats columns: [rho1, log(scv), sign*log1p(|skew|), log1p(kurt)]
        rho1_obs = feats[:, 0]
        scv_obs  = np.exp(feats[:, 1])  # undo log transform

        rho1_ok = (rho1_obs.min() - rho1_tol) <= rho1 <= (rho1_obs.max() + rho1_tol)
        scv_ok  = (scv_obs.min() / scv_factor) <= scv <= (scv_obs.max() * scv_factor)

        if rho1_ok and scv_ok:
            results.append(process_type)

    return results


def print_feasible(
    rho1: float,
    scv: float,
    skewness: float,
    kurtosis: float,
    coverage: dict | None = None,
    **kwargs,
) -> None:
    """Pretty-print feasible structures for a query point."""
    if coverage is None:
        coverage = load_coverage()

    candidates = feasible_structures(rho1, scv, skewness, kurtosis,
                                     coverage=coverage, **kwargs)
    excluded = [t for t in sm.MAP_PROCESS_TYPES if t not in candidates]

    print(f"Query: rho1={rho1:.3f}  SCV={scv:.3f}  "
          f"skewness={skewness:.3f}  kurtosis={kurtosis:.3f}\n")
    print(f"Feasible ({len(candidates)}):")
    for t in candidates:
        print(f"  + {t}")
    print(f"\nExcluded ({len(excluded)}):")
    for t in excluded:
        print(f"  - {t}")


# ---------------------------------------------------------------------------
# Entry point: build coverage dataset
# ---------------------------------------------------------------------------

_PH_TYPES = {
    "diverse_ph", "modified_renewal_ph", "sparse_ph_renewal",
    "hyper_erlang_renewal", "modified_hyper_erlang_renewal",
}


def build_coverage_tiered(
    num_fast: int = 100_000,
    num_ph: int = 10_000,
    min_size: int = 5,
    max_size: int = 50,
    seed: int = 0,
    save_path: str | Path = DEFAULT_COVERAGE_PATH,
    batch_size: int = 2_000,
) -> dict:
    """Like build_coverage but uses num_fast for non-PH types and num_ph for PH-based types."""
    from scipy.spatial import KDTree

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, np.ndarray] = {}

    for i, process_type in enumerate(sm.MAP_PROCESS_TYPES):
        num_per_type = num_ph if process_type in _PH_TYPES else num_fast
        print(f"  [{i+1:2d}/19] {process_type:<35} (n={num_per_type:,})", end="", flush=True)
        try:
            all_rows = []
            remaining = num_per_type
            batch_seed = seed + i * 1009
            while remaining > 0:
                n = min(batch_size, remaining)
                maps = sm.generate_experiment_sample(
                    process_type=process_type,
                    num_maps=n,
                    min_size=min_size,
                    max_size=max_size,
                    seed=batch_seed,
                )
                all_rows.extend(sm.compute_feature_rows(maps))
                remaining -= n
                batch_seed += 1

            if not all_rows:
                print("  no valid rows")
                continue

            feats = np.array([
                _transform(r.ac1, r.scv, r.skewness, r.kurtosis)
                for r in all_rows
            ])
            coverage[process_type] = feats
            print(f"  -> {len(all_rows):,} samples  "
                  f"rho1=[{min(r.ac1 for r in all_rows):.2f},{max(r.ac1 for r in all_rows):.2f}]  "
                  f"scv=[{min(r.scv for r in all_rows):.1f},{max(r.scv for r in all_rows):.1f}]")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    all_feats = np.concatenate(list(coverage.values()), axis=0)
    scale = all_feats.std(axis=0)
    scale[scale < 1e-8] = 1.0
    trees = {pt: KDTree(f / scale) for pt, f in coverage.items()}

    payload = {"coverage": coverage, "trees": trees, "scale": scale}
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved -> {save_path}")
    return payload


if __name__ == "__main__":
    print("Building coverage dataset (5k per type, max_size=10)...\n")
    build_coverage(num_per_type=5_000, min_size=5, max_size=10, batch_size=500)
