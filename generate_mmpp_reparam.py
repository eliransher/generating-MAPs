"""Generate MMPP via the lambda, P, Q reparametrisation and compare with the
original sampling_map method on the standard scatter plots (SCV vs rho1,
skewness vs SCV, kurtosis vs SCV).

New reparametrisation entry point
----------------------------------
Function : generate_mmpp_via_reparam(n, rng)
Location : this file (generate_mmpp_reparam.py)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUM_MAPS    = 1000
SEED        = 42
MIN_SIZE    = 2
MAX_SIZE    = 30
OUTLIER_Q   = 0.99   # percentile used for axis clipping (matching sampling_map convention)


# =============================================================================
# NEW REPARAMETRISATION
# =============================================================================

def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    Z_s = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z_s)
    return e / e.sum(axis=1, keepdims=True)


def generate_mmpp_via_reparam(n: int, rng: np.random.Generator) -> sm.MAP:
    """Generate an MMPP of size n using the lambda, P, Q reparametrisation.

    Unconstrained parameters
    ------------------------
    lambda_raw : (n,)  rates = lambda_raw^2, enforcing positivity via squaring.
    Z_raw : (n, n)  probability matrix P = softmax_rows(Z_raw).
        P_ij (i!=j) : prob of switching to state j without arrival -> D0 off-diag.
        P_ii        : prob of arrival in state i -> D1 diagonal.
    Q = I_n (fixed for MMPP — arrivals do not change state).

    Mapping to (D0, D1)
    -------------------
    D0_ij = rate_i * P_ij   (i != j)
    D0_ii = -rate_i
    D1_ii = rate_i * P_ii   (diagonal only)
    Row sums of D0+D1 = 0  [checkmark]

    Sampling strategy — match original's marginal distributions
    -----------------------------------------------------------
    The original draws arrival_rates and switching_rates INDEPENDENTLY:
        arrival_rate_i  ~ log-Uniform(0.05, 20)     [_sample_arrival_rates, sparse]
        switch_rate_ij  ~ LogNormal(-2.0, 1.0)      [_sparse_hidden_generator]

    We replicate these marginals directly:
        A_i  = arrival_rate_i ~ log-Uniform(0.05, 20)
        S_i  = per-state total switching rate ~ LogNormal(-2.0, 1.0)

    Then convert to reparam parameters:
        lambda_raw_i = sqrt(A_i + S_i)   so  lambda_raw_i^2 = A_i + S_i
        p_ii         = A_i / (A_i + S_i) = arrival fraction

    The off-diagonal P_ij (i!=j) are set via softmax of Z_raw with the
    diagonal pinned to achieve the target p_ii.  This gives the same
    (arrival rate, switching rate) marginals as the original while keeping
    the full MAP in the reparametrised form.
    """
    # --- sample arrival and switching rates matching the original's distributions ---
    # Arrival rates: log-uniform in [0.05, 20] — directly from _sample_arrival_rates(sparse)
    log_arrival = rng.uniform(np.log(0.04), np.log(25.0), size=n)

    # Switching rates: in the original, each state's total switching rate is the SUM of
    # ~(n-1)/2 lognormal(-2,1) pairwise entries, giving mean ≈ (n-1)/2 * e^{-1.5} ≈ 0.11*n.
    # We approximate this with a single lognormal whose mean matches: log_mean = -2 + log(n/2).
    log_switch_mean = -2.0 + np.log(max(n, 2) / 2.0)
    log_switch  = rng.normal(log_switch_mean, 1.0, size=n)

    arrival_rates = np.exp(log_arrival)                               # shape (n,)
    switch_rates  = np.maximum(np.exp(log_switch), 1e-8)             # shape (n,)

    # convert to reparam: lambda_raw^2 = total_rate, p_ii = arrival fraction
    total_rates = arrival_rates + switch_rates                        # lambda_raw_i^2
    p_ii        = arrival_rates / total_rates                         # arrival fraction in (0, 1)

    rates      = np.maximum(total_rates, 1e-8)                       # shape (n,)
    lambda_raw = np.sqrt(rates)                                       # satisfies positivity by squaring

    # --- build Z_raw with diagonal pinned to achieve target p_ii ---
    # with off-diagonal Z_ij ~ N(0, sigma), E[sum_{j!=i} exp(Z_ij)] = (n-1)*exp(sigma^2/2)
    # setting Z_ii = logit(p_ii) + log(n-1) + sigma^2/2 gives softmax(Z)[i,i] ≈ p_ii
    off_diag_sigma   = 1.0
    Z_raw            = rng.normal(0.0, off_diag_sigma, size=(n, n))
    log_off_sum      = np.log(n - 1) + 0.5 * off_diag_sigma ** 2
    logit_p          = np.log(p_ii / (1.0 - p_ii))
    np.fill_diagonal(Z_raw, logit_p + log_off_sum)

    # --- probability matrix P via row-wise softmax ---
    P = _softmax_rows(Z_raw)                          # shape (n, n), rows sum to 1

    # --- build D0 ---
    D0 = rates[:, None] * P                      # D0[i,j] = rate_i * P[i,j]
    np.fill_diagonal(D0, -rates)                 # D0[i,i] = -rate_i  (overwrite)

    # --- build D1 (diagonal only; Q = I) ---
    D1 = np.diag(rates * np.diag(P))            # D1[i,i] = rate_i * P[i,i]

    return sm.scale_map_to_mean(D0, D1, 1.0, "mmpp", "reparam_lambda_P_Q")


# =============================================================================
# TENSOR INTERFACE  — batch sampling of raw reparametrisation parameters
# =============================================================================

SOFTMAX_LARGE = 500.0   # M: pre-softmax value used to pin entries to ~1


def sample_mmpp_reparam_tensors(
    num_maps: int,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample unconstrained reparametrisation parameters for a batch of MMPPs.

    Parameters
    ----------
    num_maps : int
        Number of MAPs to generate.
    n : int
        Number of phases (states) in each MAP.
    seed : int or None
        Random seed.

    Returns
    -------
    lambdas : ndarray, shape (num_maps, n)
        Unsquared rate parameters.  Actual rates = lambdas**2 (positivity by
        squaring, per the reparametrisation).

    Z_P : ndarray, shape (num_maps, n, n)
        Pre-softmax unconstrained matrix for P.
        softmax_rows(Z_P)[k] gives the probability matrix of map k:
            Z_P[k, i, i]  -> arrival probability in state i  (-> D1)
            Z_P[k, i, j]  -> switching probability i->j      (-> D0 off-diag)

    Z_Q : ndarray, shape (num_maps, n, n)
        Pre-softmax unconstrained matrix for Q.
        For MMPP, Q = I_n (arrivals do not change state), encoded as:
            Z_Q[k, i, i] = M = 500   (dominates softmax -> row i stays in state i)
            Z_Q[k, i, j] = 0         (j != i)
        softmax_rows(Z_Q) ≈ I_n for any M >> log(n).
        Using M instead of literal 1/0 keeps Q in the unconstrained softmax
        space so gradient-descent optimisation leaves the identity structure
        intact: a small gradient step on Z_Q[k,i,i] barely moves softmax(Z_Q)
        away from 1 when the baseline is 500.

    Notes
    -----
    Recover (D0, D1) from the unconstrained parameters:
        rates  = lambdas**2                              # (num_maps, n)
        P      = softmax_rows(Z_P)                       # (num_maps, n, n)
        Q      = softmax_rows(Z_Q)                       # (num_maps, n, n) ≈ I
        D0[k, i, j] = rates[k, i] * P[k, i, j]         for i != j
        D0[k, i, i] = -rates[k, i]
        D1[k]       = diag(rates[k] * diag(P[k])) @ Q[k]
    """
    rng = np.random.default_rng(seed)

    log_switch_mean = -2.0 + np.log(max(n, 2) / 2.0)
    off_diag_sigma  = 1.0

    # --- sample all maps at once ---
    log_arrival = rng.uniform(np.log(0.04), np.log(25.0), size=(num_maps, n))
    log_switch  = rng.normal(log_switch_mean, 1.0, size=(num_maps, n))

    arrival_rates = np.exp(log_arrival)                              # (num_maps, n)
    switch_rates  = np.maximum(np.exp(log_switch), 1e-8)            # (num_maps, n)

    total_rates   = arrival_rates + switch_rates                     # (num_maps, n)
    p_ii          = arrival_rates / total_rates                      # (num_maps, n)

    lambda_raw    = np.sqrt(np.maximum(total_rates, 1e-8))          # (num_maps, n)

    logit_p = np.log(p_ii / (1.0 - p_ii))                          # (num_maps, n)

    # --- Z_P: pre-softmax matrix for P ---
    # Off-diagonal entries start as N(0, sigma); structural zeros from the sparse
    # hidden-chain pattern are replaced with -SOFTMAX_LARGE so that
    # softmax(Z_P)[i,j] ≈ 0 and the zero is preserved during optimisation.
    # Ring edges (i -> (i+1) % n) are always kept; up to half of the remaining
    # off-diagonal edges are randomly zeroed, mirroring _sparse_hidden_generator.
    Z_P = rng.normal(0.0, off_diag_sigma, size=(num_maps, n, n))

    idx        = np.arange(n)
    ring_cols  = (idx + 1) % n                                      # ring: i -> (i+1)%n
    ring_set   = set(zip(idx.tolist(), ring_cols.tolist()))
    candidates = [(i, j) for i in range(n) for j in range(n)
                  if i != j and (i, j) not in ring_set]

    for k in range(num_maps):
        # --- choose which candidate edges to zero for this map ---
        zero_mask = np.zeros((n, n), dtype=bool)
        if candidates:
            max_zeros  = max(1, len(candidates) // 2)
            zero_count = int(rng.integers(1, max_zeros + 1))
            chosen     = rng.choice(len(candidates), size=zero_count, replace=False)
            for ci in chosen:
                r, c = candidates[int(ci)]
                zero_mask[r, c] = True
                Z_P[k, r, c] = -SOFTMAX_LARGE   # force softmax output to ~0

        # --- pin diagonal per row, correcting for the active off-diagonal count ---
        # Formula: Z_ii = logit(p_ii) + log(active_off_diag) + sigma^2/2
        # so that softmax(Z_P)[i,i] ≈ p_ii despite the varying number of competitors.
        for i in range(n):
            active = (n - 1) - int(zero_mask[i].sum())
            log_off_sum_i = (np.log(active) + 0.5 * off_diag_sigma ** 2
                             if active > 0 else 0.0)
            Z_P[k, i, i] = logit_p[k, i] + log_off_sum_i

    # --- Z_Q: pre-softmax matrix for Q (MMPP: Q = I, encoded with M on diagonal) ---
    Z_Q = np.zeros((num_maps, n, n))
    Z_Q[:, idx, idx] = SOFTMAX_LARGE            # M=500 >> 0 -> softmax gives row ≈ e_i

    return lambda_raw, Z_P, Z_Q


# =============================================================================
# STATISTICS HELPER
# =============================================================================

def _compute_stats(map_obj: sm.MAP) -> dict | None:
    """Return {scv, skewness, kurtosis, rho1, n} or None on numerical failure."""
    try:
        moments = sm._raw_moments(map_obj.D0, map_obj.D1, max_power=4)
        stats   = sm._shape_statistics_from_raw_moments(moments)
        rho1    = sm._first_lag_correlation_from_moments(map_obj.D0, map_obj.D1, moments)
        values  = [stats["scv"], stats["skewness"], stats["kurtosis"], rho1]
        if not all(np.isfinite(v) for v in values):
            return None
        return {
            "scv":      stats["scv"],
            "skewness": stats["skewness"],
            "kurtosis": stats["kurtosis"],
            "rho1":     rho1,
            "n":        map_obj.D0.shape[0],
        }
    except Exception:
        return None


# =============================================================================
# GENERATION BATCHES
# =============================================================================

def generate_reparam_batch(num_maps: int, seed: int) -> list[dict]:
    """Generate num_maps MMPPs via generate_mmpp_via_reparam."""
    rng     = np.random.default_rng(seed)
    results = []
    attempts = 0
    while len(results) < num_maps:
        attempts += 1
        n       = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
        map_obj = generate_mmpp_via_reparam(n, rng)
        stats   = _compute_stats(map_obj)
        if stats is not None:
            results.append(stats)
        if attempts > num_maps * 20:
            print(f"  WARNING: only {len(results)} valid after {attempts} attempts")
            break
    print(f"  Reparam  : {len(results):>5} valid / {attempts} attempted  "
          f"(generate_mmpp_via_reparam  in  generate_mmpp_reparam.py)")
    return results


def generate_original_batch(num_maps: int, seed: int) -> list[dict]:
    """Generate num_maps MMPPs via sm.generate_experiment_sample (original)."""
    maps    = sm.generate_experiment_sample("mmpp", num_maps, MIN_SIZE, MAX_SIZE, seed)
    results = []
    for m in maps:
        s = _compute_stats(m)
        if s is not None:
            results.append(s)
    print(f"  Original : {len(results):>5} valid / {len(maps)} generated  "
          f"(generate_random_mmpp  in  sampling_map.py)")
    return results


# =============================================================================
# PLOTTING  (standard scatter: SCV vs rho1 / skewness vs SCV / kurtosis vs SCV)
# =============================================================================

COLOR_ORIG   = "#E64A19"   # burnt orange
COLOR_REPARM = "#1565C0"   # deep blue

def _arr(stats: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] for s in stats], dtype=float)


def _scatter_clipped(ax, x: np.ndarray, y: np.ndarray,
                     color: str, label: str, alpha: float = 0.35, s: int = 7) -> None:
    """Scatter plot clipping both axes at OUTLIER_Q percentile."""
    mask   = np.isfinite(x) & np.isfinite(y)
    x, y   = x[mask], y[mask]
    q_x    = np.quantile(x, OUTLIER_Q)
    q_y    = np.quantile(y, OUTLIER_Q)
    keep   = (x <= q_x) & (y <= q_y)
    ax.scatter(x[keep], y[keep], s=s, alpha=alpha, color=color,
               linewidths=0, rasterized=True)


def make_comparison_plots(reparam: list[dict], original: list[dict]) -> Path:
    """Produce the three standard scatter plots side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"MMPP: reparametrisation  vs  original  (n = {NUM_MAPS} each)\n"
        "Reparametrisation: generate_mmpp_via_reparam()  in  generate_mmpp_reparam.py",
        fontsize=10, fontweight="bold",
    )

    ro   = _arr(original, "rho1")
    scvo = _arr(original, "scv")
    sko  = _arr(original, "skewness")
    kuo  = _arr(original, "kurtosis")

    rr   = _arr(reparam, "rho1")
    scvr = _arr(reparam, "scv")
    skr  = _arr(reparam, "skewness")
    kur  = _arr(reparam, "kurtosis")

    # --- panel 1: SCV vs rho1 ---
    ax = axes[0]
    _scatter_clipped(ax, ro,   scvo, COLOR_ORIG,   "original",         alpha=0.40)
    _scatter_clipped(ax, rr,   scvr, COLOR_REPARM, "reparam (λ, P, Q)", alpha=0.40)
    ax.set_xlabel("First-lag autocorrelation  rho1")
    ax.set_ylabel("SCV")
    ax.set_title("SCV  vs  rho1")
    ax.axhline(1, color="gray", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.6, ls="--", zorder=0)

    # --- panel 2: Skewness vs SCV ---
    ax = axes[1]
    _scatter_clipped(ax, scvo, sko,  COLOR_ORIG,   "original")
    _scatter_clipped(ax, scvr, skr,  COLOR_REPARM, "reparam (λ, P, Q)")
    ax.set_xlabel("SCV")
    ax.set_ylabel("Skewness")
    ax.set_title("Skewness  vs  SCV")

    # --- panel 3: Kurtosis vs SCV ---
    ax = axes[2]
    _scatter_clipped(ax, scvo, kuo,  COLOR_ORIG,   "original")
    _scatter_clipped(ax, scvr, kur,  COLOR_REPARM, "reparam (λ, P, Q)")
    ax.set_xlabel("SCV")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis  vs  SCV")

    # shared legend
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ORIG,   marker="o", ls="None",
                      markersize=6, label=f"original  (sampling_map.generate_random_mmpp)  n={len(original)}"),
        mlines.Line2D([], [], color=COLOR_REPARM, marker="o", ls="None",
                      markersize=6, label=f"reparam  (generate_mmpp_via_reparam)  n={len(reparam)}"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = PLOT_DIR / "mmpp_reparam_vs_original.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    lams, ps, qs  = sample_mmpp_reparam_tensors(num_maps =2, n=3, seed=SEED)  # quick test of tensor interface


    print("=" * 65)
    print("MMPP generation: reparametrisation vs original")
    print("=" * 65)
    print(f"\nGenerating {NUM_MAPS} MMPPs via lambda, P, Q reparametrisation ...")
    reparam = generate_reparam_batch(NUM_MAPS, SEED)

    print(f"\nGenerating {NUM_MAPS} MMPPs via original sampling_map ...")
    original = generate_original_batch(NUM_MAPS, SEED + 1)

    print("\nPlotting ...")
    out = make_comparison_plots(reparam, original)
    print(f"  Saved -> {out}")

    # quick summary table
    print("\n" + "-" * 55)
    print(f"{'metric':<12}  {'reparam  min':>12}  {'reparam  max':>12}  "
          f"{'original min':>12}  {'original max':>12}")
    print("-" * 55)
    for key in ("rho1", "scv", "skewness", "kurtosis"):
        rv = np.array([s[key] for s in reparam])
        ov = np.array([s[key] for s in original])
        print(f"{key:<12}  {rv.min():>12.3f}  {rv.max():>12.3f}  "
              f"{ov.min():>12.3f}  {ov.max():>12.3f}")
    print("=" * 65)
    pass

if __name__ == "__main__":
    main()
