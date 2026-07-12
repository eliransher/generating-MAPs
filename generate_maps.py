"""Generate 10 MAPs from each type, compute moments and polynomial autocorrelations, save to data/.

Output: data/{process_type}_maps.pkl
Each file: list of dicts with keys:
  D0, D1,
  m_1 .. m_5  (raw moments of inter-arrival time)
  rho_i_j_k   (8 values: i,j in {1,2}, k in {1,2})
  process_type, structure, sample_id  (metadata)

PH-based types (diverse_ph, modified_renewal_ph, sparse_ph_renewal,
hyper_erlang_renewal, modified_hyper_erlang_renewal) normally rely on a
torch-dependent PH sampler.  When torch is unavailable the script falls back
to pure-numpy PH generators that produce equivalent MAP structures.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add the code directory so we can import sampling_map (and sampling__PH if
# torch is available)
# ---------------------------------------------------------------------------
CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))

import sampling_map as sm

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_MAPS = 10
BASE_SEED = 42

# (i, j, k) tuples for the 8 polynomial autocorrelations
POLY_AC_PARAMS = [
    (1, 1, 1), (1, 1, 2),
    (1, 2, 1), (1, 2, 2),
    (2, 1, 1), (2, 1, 2),
    (2, 2, 1), (2, 2, 2),
]

# erlang_rate_spread needs k_modes*m_phases <= max_size, with m_phases >= 5.
# k_modes can reach 7, so min total = 35 -> use 70 to give plenty of room.
PER_TYPE_MAX_SIZE: dict[str, int] = {
    "erlang_rate_spread": 70,
}

# ---------------------------------------------------------------------------
# Fallback pure-numpy PH-based MAP generators (no torch required)
# ---------------------------------------------------------------------------

def _ph_renewal_map(n: int, ph_structure: str, rng: np.random.Generator) -> sm.MAP:
    """Renewal MAP built from a simple PH subgenerator (no torch required).

    Steps:
      1. Build a hidden CTMC generator Q of size n (rows sum to 0).
      2. Add absorption: subtract a random fraction of each diagonal to create
         exit rates t_i = frac_i * |Q_{ii}|, making T = Q with diagonal shifted.
      3. D0 = T,  D1 = outer(t, alpha)  (renewal: restarts from alpha).
    """
    if ph_structure in ("coxian_ph", "general_ph"):
        structure = "coxian" if ph_structure == "coxian_ph" else "sparse"
    else:  # hyper_erlang_ph, modified_hyper_erlang_ph
        structure = "hyper_erlang"

    Q = sm._build_hidden_generator(n, structure, rng)
    diag_abs = np.abs(np.diag(Q))                      # |Q_{ii}|
    frac = rng.uniform(0.05, 0.40, size=n)
    exit_rates = diag_abs * frac
    exit_rates = np.maximum(exit_rates, 1e-10)

    T = Q.copy()
    np.fill_diagonal(T, np.diag(Q) - exit_rates)       # rows now sum to -exit_rate

    alpha = rng.dirichlet(np.ones(n))
    D0 = T
    D1 = np.outer(exit_rates, alpha)
    return sm.scale_map_to_mean(D0, D1, 1.0, "diverse_ph", ph_structure)


def _modified_ph_renewal_map(
    n: int,
    ph_structure: str,
    correlation_sign: int,
    rng: np.random.Generator,
) -> sm.MAP:
    """PH MAP with targeted (correlated) restart, no torch required."""
    if ph_structure in ("coxian_ph", "general_ph"):
        structure = "coxian" if ph_structure == "coxian_ph" else "sparse"
    else:
        structure = "hyper_erlang"

    Q = sm._build_hidden_generator(n, structure, rng)
    diag_abs = np.abs(np.diag(Q))
    frac = rng.uniform(0.05, 0.40, size=n)
    exit_rates = np.maximum(diag_abs * frac, 1e-10)

    T = Q.copy()
    np.fill_diagonal(T, np.diag(Q) - exit_rates)
    D0 = T

    alpha = rng.dirichlet(np.ones(n))
    D1 = sm._targeted_restart_arrival_matrix(D0, exit_rates, alpha, correlation_sign, rng)
    label = "positive" if correlation_sign > 0 else "negative"
    return sm.scale_map_to_mean(D0, D1, 1.0, "modified_renewal_ph", f"targeted_{label}_{ph_structure}")


PH_STRUCTURES_CYCLE = ("coxian_ph", "hyper_erlang_ph", "general_ph", "modified_hyper_erlang_ph")


def _generate_ph_based(
    process_type: str,
    num_maps: int,
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
) -> list[sm.MAP]:
    maps: list[sm.MAP] = []
    n_structs = len(PH_STRUCTURES_CYCLE)
    for idx in range(num_maps):
        ph_structure = PH_STRUCTURES_CYCLE[idx % n_structs]
        n = int(rng.integers(max(min_size, 4), max_size + 1))
        seed_inner = rng  # reuse same rng for reproducibility

        if process_type in ("diverse_ph", "sparse_ph_renewal", "hyper_erlang_renewal",
                            "modified_hyper_erlang_renewal"):
            if process_type == "sparse_ph_renewal":
                ph_structure = "general_ph"
            elif process_type == "hyper_erlang_renewal":
                ph_structure = "hyper_erlang_ph"
            elif process_type == "modified_hyper_erlang_renewal":
                ph_structure = "modified_hyper_erlang_ph"
            maps.append(_ph_renewal_map(n, ph_structure, rng))

        elif process_type == "modified_renewal_ph":
            correlation_sign = 1 if idx % 2 == 0 else -1
            maps.append(_modified_ph_renewal_map(n, ph_structure, correlation_sign, rng))

    return maps


# ---------------------------------------------------------------------------
# Core record builder
# ---------------------------------------------------------------------------

def compute_map_record(map_obj: sm.MAP, sample_id: int) -> dict | None:
    D0, D1 = map_obj.D0, map_obj.D1
    try:
        raw = sm._raw_moments(D0, D1, max_power=5)
        # raw[0] = 1 (0th moment), raw[1]=m_1, ..., raw[5]=m_5
        moments = {f"m_{k}": raw[k] for k in range(1, 6)}

        correlations: dict[str, float] = {}
        for i, j, k in POLY_AC_PARAMS:
            key = f"rho_{i}_{j}_{k}"
            correlations[key] = sm.kth_lag_power_autocorrelation(D0, D1, k, i, j)

        if not all(np.isfinite(v) for v in moments.values()):
            return None
        if not all(np.isfinite(v) for v in correlations.values()):
            return None

        return {
            "D0": D0,
            "D1": D1,
            **moments,
            **correlations,
            "process_type": map_obj.process_type,
            "structure": map_obj.structure,
            "sample_id": sample_id,
        }
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Per-type generation
# ---------------------------------------------------------------------------

PH_TYPES = frozenset({
    "diverse_ph", "modified_renewal_ph",
    "sparse_ph_renewal", "hyper_erlang_renewal", "modified_hyper_erlang_renewal",
})


def generate_and_save(process_type: str, seed: int) -> None:
    min_size = 2
    max_size = PER_TYPE_MAX_SIZE.get(process_type, 30)
    rng = np.random.default_rng(seed)

    print(f"  Generating {NUM_MAPS} MAPs for '{process_type}' (max_size={max_size}) ...")

    if process_type in PH_TYPES:
        # Try the original generator first (works when torch is installed)
        try:
            maps = sm.generate_experiment_sample(
                process_type=process_type,
                num_maps=NUM_MAPS,
                min_size=min_size,
                max_size=max_size,
                seed=seed,
            )
        except (ModuleNotFoundError, ImportError):
            print(f"    torch unavailable — using pure-numpy PH fallback")
            maps = _generate_ph_based(process_type, NUM_MAPS, min_size, max_size, rng)
    else:
        maps = sm.generate_experiment_sample(
            process_type=process_type,
            num_maps=NUM_MAPS,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
        )

    records: list[dict] = []
    for sample_id, map_obj in enumerate(maps):
        record = compute_map_record(map_obj, sample_id)
        if record is not None:
            records.append(record)
        else:
            print(f"    sample {sample_id}: skipped (numerical issue)")

    out_path = OUTPUT_DIR / f"{process_type}_maps.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"    saved {len(records)} records -> {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"MAP types: {len(sm.MAP_PROCESS_TYPES)}, samples each: {NUM_MAPS}\n")

    for idx, process_type in enumerate(sm.MAP_PROCESS_TYPES):
        type_seed = BASE_SEED + idx * 1009
        try:
            generate_and_save(process_type, type_seed)
        except Exception as exc:
            print(f"  ERROR for '{process_type}': {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
