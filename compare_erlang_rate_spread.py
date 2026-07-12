"""Compare erlang_rate_spread: original sampler vs reparam sampler.

Three standard scatter plots:
  1. SCV vs rho1
  2. Skewness vs SCV
  3. Kurtosis vs SCV
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm
from generate_map_reparam import sample_erlang_rate_spread_reparam_tensors

PLOT_DIR   = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUM_MAPS  = 5_000
N         = 10          # fixed MAP size for both methods
SEED      = 42
OUTLIER_Q = 0.99

COLOR_ORIG   = "#E64A19"
COLOR_REPARM = "#1565C0"


# ---------------------------------------------------------------------------
# Reconstruct (D0, D1) from a single reparam tensor triple
# ---------------------------------------------------------------------------

def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    Z_s = Z - Z.max(axis=-1, keepdims=True)
    e   = np.exp(Z_s)
    return e / e.sum(axis=-1, keepdims=True)


def _reparam_to_map(lambda_raw: np.ndarray,
                    Z_P: np.ndarray,
                    Z_Q: np.ndarray) -> sm.MAP:
    """Convert a single (lambda_raw, Z_P, Z_Q) triple to a MAP object."""
    rates = lambda_raw ** 2
    P     = _softmax_rows(Z_P)
    Q     = _softmax_rows(Z_Q)
    n     = len(rates)

    D0 = rates[:, None] * P
    np.fill_diagonal(D0, -rates)

    D1 = np.diag(rates * np.diag(P)) @ Q

    return sm.scale_map_to_mean(D0, D1, 1.0,
                                "erlang_rate_spread", "reparam")


# ---------------------------------------------------------------------------
# Stats helper (same approach as generate_mmpp_reparam.py)
# ---------------------------------------------------------------------------

def _compute_stats(map_obj: sm.MAP) -> dict | None:
    try:
        moments = sm._raw_moments(map_obj.D0, map_obj.D1, max_power=4)
        stats   = sm._shape_statistics_from_raw_moments(moments)
        rho1    = sm._first_lag_correlation_from_moments(
                      map_obj.D0, map_obj.D1, moments)
        values  = [stats["scv"], stats["skewness"], stats["kurtosis"], rho1]
        if not all(np.isfinite(v) for v in values):
            return None
        return {
            "scv":      stats["scv"],
            "skewness": stats["skewness"],
            "kurtosis": stats["kurtosis"],
            "rho1":     rho1,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generation batches
# ---------------------------------------------------------------------------

def generate_original_batch(num_maps: int, seed: int) -> list[dict]:
    # Match reparam exactly: k_modes=2, m_phases=N//2, same parameter distributions
    rng = np.random.default_rng(seed)
    k_modes, m_phases = 2, max(1, N // 2)
    maps = []
    for _ in range(num_maps):
        rs   = float(np.exp(rng.uniform(np.log(50.0), np.log(5000.0))))
        conc = float(np.exp(rng.uniform(np.log(5.0),  np.log(100.0))))
        maps.append(sm.generate_erlang_rate_spread_map(
            k_modes, m_phases, rate_spread=rs,
            same_mode_concentration=conc,
            seed=int(rng.integers(0, 2**31)),
        ))
    results = [s for m in maps if (s := _compute_stats(m)) is not None]
    print(f"  Original : {len(results):>5} valid / {len(maps)} generated  "
          f"(n={N}, k_modes={k_modes}, m_phases={m_phases})")
    return results


def generate_reparam_batch(num_maps: int, seed: int) -> list[dict]:
    lambda_raws, Z_Ps, Z_Qs = sample_erlang_rate_spread_reparam_tensors(
        num_maps, N, seed=seed)
    results = []
    for k in range(num_maps):
        m = _reparam_to_map(lambda_raws[k], Z_Ps[k], Z_Qs[k])
        s = _compute_stats(m)
        if s is not None:
            results.append(s)
    print(f"  Reparam  : {len(results):>5} valid / {num_maps} generated  (n={N})")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _arr(stats: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] for s in stats], dtype=float)


def _scatter_clipped(ax, x, y, color, label, alpha=0.35, s=6):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    qx = np.quantile(x, OUTLIER_Q)
    qy = np.quantile(y, OUTLIER_Q)
    keep = (x <= qx) & (y <= qy)
    ax.scatter(x[keep], y[keep], s=s, alpha=alpha, color=color,
               linewidths=0, rasterized=True, label=label)


def make_plots(orig: list[dict], reparam: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"erlang_rate_spread: original vs reparam  ({NUM_MAPS} samples each, n={N})",
        fontsize=10, fontweight="bold",
    )

    ro, scvo, sko, kuo = (_arr(orig, k) for k in ("rho1", "scv", "skewness", "kurtosis"))
    rr, scvr, skr, kur = (_arr(reparam, k) for k in ("rho1", "scv", "skewness", "kurtosis"))

    ax = axes[0]
    _scatter_clipped(ax, ro,   scvo, COLOR_ORIG,   "original")
    _scatter_clipped(ax, rr,   scvr, COLOR_REPARM, "reparam")
    ax.set_xlabel("rho1"); ax.set_ylabel("SCV")
    ax.set_title("SCV vs rho1")
    ax.axhline(1, color="gray", lw=0.6, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.6, ls="--", zorder=0)

    ax = axes[1]
    _scatter_clipped(ax, scvo, sko,  COLOR_ORIG,   "original")
    _scatter_clipped(ax, scvr, skr,  COLOR_REPARM, "reparam")
    ax.set_xlabel("SCV"); ax.set_ylabel("Skewness")
    ax.set_title("Skewness vs SCV")

    ax = axes[2]
    _scatter_clipped(ax, scvo, kuo,  COLOR_ORIG,   "original")
    _scatter_clipped(ax, scvr, kur,  COLOR_REPARM, "reparam")
    ax.set_xlabel("SCV"); ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis vs SCV")

    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ORIG,   marker="o", ls="None",
                      markersize=6, label=f"original  ({len(orig)} samples)"),
        mlines.Line2D([], [], color=COLOR_REPARM, marker="o", ls="None",
                      markersize=6, label=f"reparam  ({len(reparam)} samples)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = PLOT_DIR / "erlang_rate_spread_reparam_vs_original.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("erlang_rate_spread: original vs reparam")
    print("=" * 60)

    print(f"\nGenerating {NUM_MAPS} original maps ...")
    orig   = generate_original_batch(NUM_MAPS, SEED)

    print(f"\nGenerating {NUM_MAPS} reparam maps ...")
    reparam = generate_reparam_batch(NUM_MAPS, SEED + 1)

    print("\nPlotting ...")
    out = make_plots(orig, reparam)
    print(f"\n  Plot saved to:\n    {out.resolve()}")

    print("\n" + "-" * 60)
    print(f"{'metric':<12}  {'orig min':>10}  {'orig max':>10}  "
          f"{'rep min':>10}  {'rep max':>10}")
    print("-" * 60)
    for key in ("rho1", "scv", "skewness", "kurtosis"):
        ov = _arr(orig, key)
        rv = _arr(reparam, key)
        print(f"{key:<12}  {ov.min():>10.3f}  {ov.max():>10.3f}  "
              f"{rv.min():>10.3f}  {rv.max():>10.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
