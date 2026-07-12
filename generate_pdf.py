"""Generate a clean PDF document describing all 19 MAP structure types,
including D0/D1 sparsity pattern visualisations."""

import sys
from pathlib import Path

import numpy as np
from fpdf import FPDF, XPos, YPos

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm

OUT_PATH = Path(__file__).parent / "MAP_structures.pdf"

TITLE = "Markovian Arrival Process (MAP) Structure Types"
SUBTITLE = "A Reference Guide to the 19 Supported MAP Families"

# ---------------------------------------------------------------------------
# Small example MAP generators (n <= 8)
# ---------------------------------------------------------------------------

def _ph_renewal_example(structure: str, n: int, rng: np.random.Generator) -> sm.MAP:
    Q = sm._build_hidden_generator(n, structure, rng)
    diag_abs = np.abs(np.diag(Q))
    frac = rng.uniform(0.2, 0.5, size=n)
    exit_rates = np.maximum(diag_abs * frac, 1e-10)
    T = Q.copy()
    np.fill_diagonal(T, np.diag(Q) - exit_rates)
    alpha = rng.dirichlet(np.ones(n))
    D1 = np.outer(exit_rates, alpha)
    return sm.scale_map_to_mean(T, D1, 1.0, "ph_renewal", structure)


def _modified_ph_example(n: int, rng: np.random.Generator) -> sm.MAP:
    Q = sm._build_hidden_generator(n, "coxian", rng)
    diag_abs = np.abs(np.diag(Q))
    exit_rates = np.maximum(diag_abs * rng.uniform(0.2, 0.5, size=n), 1e-10)
    T = Q.copy()
    np.fill_diagonal(T, np.diag(Q) - exit_rates)
    alpha = rng.dirichlet(np.ones(n))
    D1 = sm._targeted_restart_arrival_matrix(T, exit_rates, alpha, 1, rng)
    return sm.scale_map_to_mean(T, D1, 1.0, "modified_renewal_ph", "coxian")


def get_example_map(process_type: str) -> sm.MAP | None:
    rng = np.random.default_rng(42)
    try:
        if process_type == "mmpp":
            return sm.generate_random_mmpp(4, "sparse", 42)
        if process_type == "alternating_blocks":
            return sm.generate_alternating_blocks_map(2, 2, "sparse", "coxian", 42)
        if process_type == "alternating_extreme":
            return sm.generate_alternating_extreme_map(3, 3, 42)
        if process_type == "alternating_extreme_hyper":
            return sm.generate_alternating_extreme_hyper_map(4, 4, 42)
        if process_type == "diverse_ph":
            return _ph_renewal_example("coxian", 4, rng)
        if process_type == "modified_renewal_ph":
            return _modified_ph_example(4, rng)
        if process_type == "random_map":
            return sm.generate_random_map_method_map(4, 42)
        if process_type == "random_map_negative1":
            return sm.generate_random_map_negative1_method_map(4, 42)
        if process_type == "erlang_renewal":
            return sm.generate_erlang_renewal_map(4, 42)
        if process_type == "sparse_ph_renewal":
            return _ph_renewal_example("sparse", 4, rng)
        if process_type == "hyper_erlang_renewal":
            return _ph_renewal_example("hyper_erlang", 6, rng)
        if process_type == "modified_hyper_erlang_renewal":
            return _ph_renewal_example("hyper_erlang", 6, rng)
        if process_type == "slow_switching_mmpp":
            return sm.generate_slow_switching_mmpp(4, 0.02, 2.5, 42)
        if process_type == "erlang_modulated":
            return sm.generate_erlang_modulated_map(2, 2, 8.0, 42)
        if process_type == "hyper_telegraph":
            return sm.generate_hyper_telegraph_map(2, 2, (5.0, 50.0), 1.5, 42)
        if process_type == "erlang_telegraph":
            return sm.generate_erlang_telegraph_map(2, 2, 2.0, 42)
        if process_type == "erlang_rate_spread":
            return sm.generate_erlang_rate_spread_map(2, 3, 200.0, 10.0, 42)
        if process_type == "coxian_alternating":
            return sm.generate_coxian_alternating_map(3, 3, 5.0, 1.5, 42)
        if process_type == "erlang_hyper_telegraph":
            return sm.generate_erlang_hyper_telegraph_map(3, 2, 50.0, 5.0, 42)
    except Exception as exc:
        print(f"  Warning: example generation failed for '{process_type}': {exc}")
    return None


# ---------------------------------------------------------------------------
# Matrix structure drawing
# ---------------------------------------------------------------------------

# Colours
COL_D0_NZ   = (70,  120, 200)   # blue  – D0 off-diagonal non-zero
COL_D0_DIAG = (30,  60,  140)   # dark navy – D0 diagonal (always non-zero)
COL_D1_NZ   = (50,  170, 90)    # green – D1 non-zero
COL_ZERO    = (235, 235, 235)   # light gray – zero entry
COL_BORDER  = (160, 160, 160)


def _draw_single_matrix(
    pdf: FPDF,
    mat: np.ndarray,
    label: str,
    x0: float,
    y0: float,
    cs: float,
    nz_color: tuple,
    is_D0: bool = False,
) -> None:
    """Draw the sparsity grid for one matrix at (x0, y0)."""
    n = mat.shape[0]
    idx_offset = cs * 0.9  # space for row/col indices

    # Matrix label centred above the grid
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(x0 + idx_offset, y0)
    pdf.cell(n * cs, 6, label, align="C")

    # Column indices
    pdf.set_font("Helvetica", "", 6)
    pdf.set_text_color(100, 100, 100)
    for j in range(n):
        pdf.set_xy(x0 + idx_offset + j * cs, y0 + 6)
        pdf.cell(cs, 4, str(j), align="C")

    grid_y = y0 + 10

    for i in range(n):
        # Row index
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(x0, grid_y + i * cs + (cs - 4) / 2)
        pdf.cell(idx_offset, 4, str(i), align="R")

        for j in range(n):
            cx = x0 + idx_offset + j * cs
            cy = grid_y + i * cs
            val = mat[i, j]
            is_nz = abs(val) > 1e-10

            if is_nz:
                color = COL_D0_DIAG if (is_D0 and i == j) else nz_color
            else:
                color = COL_ZERO

            pdf.set_fill_color(*color)
            pdf.set_draw_color(*COL_BORDER)
            pdf.set_line_width(0.15)
            pdf.rect(cx, cy, cs, cs, style="FD")


def draw_matrix_pair(pdf: FPDF, D0: np.ndarray, D1: np.ndarray) -> None:
    """Draw D0 and D1 sparsity grids side-by-side and advance the y cursor."""
    n = D0.shape[0]
    cs = min(5.5, 38.0 / n)          # cell size: shrink for large n
    idx_offset = cs * 0.9
    mat_w = idx_offset + n * cs
    gap = 12.0

    avail = pdf.w - pdf.l_margin - pdf.r_margin
    total_w = 2 * mat_w + gap
    x_start = pdf.l_margin + max(0, (avail - total_w) / 2)

    y = pdf.get_y()
    grid_h = 10 + n * cs          # header + col-indices + grid

    _draw_single_matrix(pdf, D0, "D0", x_start,              y, cs, COL_D0_NZ, is_D0=True)
    _draw_single_matrix(pdf, D1, "D1", x_start + mat_w + gap, y, cs, COL_D1_NZ, is_D0=False)

    # Legend
    legend_y = y + grid_h + 1
    pdf.set_xy(x_start + idx_offset, legend_y)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(80, 80, 80)

    def _legend_box(r, g, b, text):
        lx = pdf.get_x()
        ly = pdf.get_y()
        pdf.set_fill_color(r, g, b)
        pdf.set_draw_color(*COL_BORDER)
        pdf.set_line_width(0.15)
        pdf.rect(lx, ly + 0.5, 4, 4, style="FD")
        pdf.set_xy(lx + 5, ly)
        pdf.cell(28, 5, text)

    _legend_box(*COL_D0_DIAG, "D0 diagonal")
    _legend_box(*COL_D0_NZ,   "D0 off-diag nz")
    _legend_box(*COL_D1_NZ,   "D1 non-zero")
    _legend_box(*COL_ZERO,    "zero")

    pdf.set_y(legend_y + 7)


# ---------------------------------------------------------------------------
# MAP type catalogue
# ---------------------------------------------------------------------------

MAP_TYPES = [
    (
        "mmpp",
        "Standard",
        "Moderate (depends on structure)",
        ">= 1",
        (
            "Markov-Modulated Poisson Process. The background state evolves as a "
            "continuous-time Markov chain (CTMC), and arrivals occur at a Poisson rate "
            "that depends on the current state. D1 is diagonal, meaning arrivals do not "
            "trigger a state transition."
        ),
        (
            "D0 = Q - diag(lambda): sparse/Coxian/hyper-Erlang generator minus arrival "
            "rates on the diagonal. Off-diagonal entries of D0 are a subset of Q. "
            "D1 = diag(lambda): diagonal matrix of per-state arrival rates; all "
            "off-diagonal entries are zero."
        ),
    ),
    (
        "alternating_blocks",
        "Standard",
        "Typically negative (block alternation)",
        "Variable",
        (
            "Two-block MAP where each observed arrival switches the process to the "
            "first phase of the opposite block. Block A is the slow block and block B "
            "the fast block. Each block independently uses a sparse, Coxian, or "
            "hyper-Erlang internal structure."
        ),
        (
            "D0 is block-diagonal: upper-left (block A) and lower-right (block B) "
            "are dense, all cross-block entries are zero. D1 is block off-diagonal: "
            "upper-right (A arrivals enter B) and lower-left (B arrivals enter A) "
            "are non-zero; the diagonal blocks of D1 are zero."
        ),
    ),
    (
        "alternating_extreme",
        "Standard",
        "Strongly negative (approaching -1 with extreme ratio)",
        "Approx. 1 (low to moderate)",
        (
            "Extreme slow/fast alternating MAP. Each block is an Erlang chain: the "
            "process advances through phases without arrivals, and only the final phase "
            "emits an arrival that starts the opposite block. The slow block has mean "
            "8-80x longer than the fast block, creating very strong negative first-lag "
            "autocorrelation."
        ),
        (
            "D0 is block-tridiagonal: within each Erlang chain, only the superdiagonal "
            "is non-zero (forward phase progression). All cross-block entries of D0 are "
            "zero. D1 has exactly two non-zero entries: D1[last_A, first_B] and "
            "D1[last_B, first_A]. All other D1 entries are zero."
        ),
    ),
    (
        "alternating_extreme_hyper",
        "Standard",
        "Negative (preserved from alternation)",
        "High (hyper-Erlang branches widen SCV)",
        (
            "Strict slow/fast alternation with hyper-Erlang branch variability. Like "
            "alternating_extreme but each block consists of multiple Erlang branches "
            "with different means. This widens the SCV, skewness, and kurtosis while "
            "preserving the negative autocorrelation from the alternation structure."
        ),
        (
            "D0 is block-diagonal with each block having an internal hyper-Erlang "
            "structure (Erlang chains in parallel). D1 is block off-diagonal: only "
            "the last phase of each branch in block A can trigger an arrival into "
            "block B, and vice versa. Most entries of D1 are zero."
        ),
    ),
    (
        "diverse_ph",
        "Renewal (rho1 = 0)",
        "0 (renewal process)",
        "Variable",
        (
            "Renewal MAP built from a phase-type (PH) distribution. Since each "
            "inter-arrival time is drawn independently from the same PH distribution, "
            "consecutive inter-arrivals are uncorrelated (rho1 = 0 exactly). Covers "
            "four PH sub-structures: Coxian, Hyper-Erlang, General (sparse), and "
            "Modified Hyper-Erlang."
        ),
        (
            "D0 = T (PH subgenerator): sparsity depends on the PH type (Coxian = "
            "upper-triangular with resets; hyper-Erlang = block structure). D1 has "
            "rank 1: D1 = outer(t, alpha), so every row of D1 that corresponds to "
            "a state with a non-zero exit rate t_i is entirely non-zero, and rows "
            "with t_i = 0 are entirely zero."
        ),
    ),
    (
        "modified_renewal_ph",
        "Standard",
        "Positive or negative (targeted restart)",
        "Variable (PH-dependent)",
        (
            "PH-based MAP with a targeted (non-renewal) arrival restart distribution "
            "in D1. Unlike diverse_ph, the restart probabilities after an arrival are "
            "biased to induce positive or negative inter-arrival dependence. The sign "
            "of the correlation is controlled explicitly."
        ),
        (
            "D0 = T (PH subgenerator): same sparse structure as diverse_ph. D1 is "
            "no longer rank-1: each row of D1 has its own non-zero pattern, biased so "
            "that exits from fast states preferentially restart in fast states (positive "
            "correlation) or slow states (negative correlation). Most rows remain dense."
        ),
    ),
    (
        "random_map",
        "Standard",
        "Random (no control)",
        "Random",
        (
            "Fully random MAP generated by the random_map() helper. Both D0 and D1 "
            "are constructed from random non-negative matrices subject to MAP validity "
            "constraints (D0 + D1 is a valid generator). Provides unstructured "
            "baseline coverage of the (rho1, SCV) space."
        ),
        (
            "D0 has non-positive diagonal and non-negative off-diagonal entries; "
            "D1 has non-negative entries. Both matrices are generally dense: almost "
            "all entries are non-zero. D0 + D1 sums to a valid CTMC generator "
            "(rows sum to 0)."
        ),
    ),
    (
        "random_map_negative1",
        "Standard",
        "Biased negative",
        "Random",
        (
            "Random MAP generated by the random_map_negative1() helper, which is "
            "biased to produce negative first-lag autocorrelation. Provides "
            "unstructured baseline coverage of the negative-rho1 region."
        ),
        (
            "Similar matrix structure to random_map: both D0 and D1 are generally "
            "dense. The construction biases D1 toward off-diagonal dominant entries, "
            "encouraging state alternation after each arrival and thus negative "
            "autocorrelation, but the exact sparsity pattern is random."
        ),
    ),
    (
        "erlang_renewal",
        "Renewal (rho1 = 0)",
        "0 (renewal process)",
        "1/k (low for large k)",
        (
            "Erlang-k renewal MAP. Each inter-arrival time is the sum of k i.i.d. "
            "exponential phases, giving SCV = 1/k. For k=1 this reduces to a Poisson "
            "process (SCV = 1). As k increases, inter-arrivals become increasingly "
            "deterministic. All consecutive inter-arrivals are independent (rho1 = 0)."
        ),
        (
            "D0 is tridiagonal: diagonal entries are -rate and superdiagonal entries "
            "are +rate (phase progression). All other D0 entries are zero. D1 has "
            "exactly one non-zero entry: D1[k-1, 0] = rate (last phase triggers "
            "arrival and restarts at phase 0). All other D1 entries are zero."
        ),
    ),
    (
        "sparse_ph_renewal",
        "Renewal (rho1 = 0)",
        "0 (renewal process)",
        "Variable (general PH)",
        (
            "Renewal MAP from a sparse general PH distribution. The PH subgenerator "
            "has many zero off-diagonal entries (only a guaranteed ring structure plus "
            "random non-zero edges). This gives a varied SCV range depending on the "
            "sampled PH. Inter-arrivals are always independent (rho1 = 0 exactly)."
        ),
        (
            "D0 = T (sparse PH subgenerator): has a guaranteed ring of forward "
            "transitions plus a random subset of additional off-diagonal entries; "
            "most off-diagonal entries are zero. D1 = outer(t, alpha): same rank-1 "
            "structure as diverse_ph. Non-zero rows of D1 correspond to states with "
            "non-zero exit rates."
        ),
    ),
    (
        "hyper_erlang_renewal",
        "Renewal (rho1 = 0)",
        "0 (renewal process)",
        "High (multiple Erlang branches)",
        (
            "Renewal MAP from a hyper-Erlang PH distribution. Multiple parallel "
            "Erlang chains with different rates are entered with probabilities drawn "
            "from a Dirichlet. This can achieve high SCV (>> 1) while remaining a "
            "renewal process (rho1 = 0 exactly)."
        ),
        (
            "D0 is block-structured: within each Erlang branch, only the superdiagonal "
            "is non-zero (phase progression); there are no cross-branch entries in D0. "
            "D1 is sparse: only the last phase of each branch has non-zero entries "
            "(the exit phases), and each such row is fully non-zero (rank-1 structure)."
        ),
    ),
    (
        "modified_hyper_erlang_renewal",
        "Renewal (rho1 = 0)",
        "0 (renewal process)",
        "High (wider than standard hyper-Erlang)",
        (
            "Renewal MAP from a modified hyper-Erlang PH, where the restart "
            "distribution within each branch end is random (not simply returning to "
            "the branch start). This gives wider coverage of skewness and kurtosis "
            "than standard hyper-Erlang while keeping rho1 = 0 exactly."
        ),
        (
            "D0 has the same hyper-Erlang block structure as hyper_erlang_renewal: "
            "within-branch superdiagonal only, no cross-branch entries. D1 has the "
            "same rank-1 outer-product structure, so non-zero rows appear at branch "
            "exits. The difference lies in the random entry distribution within those "
            "rows, not in the sparsity pattern."
        ),
    ),
    (
        "slow_switching_mmpp",
        "Zone A: strong positive rho1",
        "0.5 to 0.95 (very positive)",
        "Moderate to high",
        (
            "MMPP with a slow background Markov chain. The switching rate between "
            "states is scaled down by a factor (0.001-0.15) relative to the arrival "
            "rates. Because the state persists across many consecutive arrivals, "
            "consecutive inter-arrival times are highly correlated. Rate spread "
            "across states independently controls SCV."
        ),
        (
            "D1 = diag(lambda): diagonal only, identical structure to a standard "
            "MMPP. D0 = Q_slow - diag(lambda): the off-diagonal entries of D0 come "
            "from Q_slow (sparse CTMC scaled by switching_scale), all very small in "
            "magnitude relative to the diagonal. Sparsity pattern is the same as "
            "mmpp; only the magnitude ratio differs."
        ),
    ),
    (
        "erlang_modulated",
        "Zone B: positive rho1, low SCV",
        "> 0 (0.3-0.7 typical)",
        "< 1 (low, Erlang-controlled)",
        (
            "k parallel Erlang chains, each with its own rate, and biased same-chain "
            "restart probabilities after each arrival. The Erlang phases keep SCV low "
            "(approx. 1/m_phases), while the same-chain restart bias creates positive "
            "autocorrelation. Fills the (rho1 > 0, SCV < 1) region that MMPPs and "
            "renewal processes cannot cover."
        ),
        (
            "D0 is block-diagonal: within each Erlang chain, only the superdiagonal "
            "is non-zero (phase progression), all cross-chain entries are zero. D1 "
            "is sparse with a specific pattern: only the last phase of each chain "
            "can trigger an arrival; that row of D1 has non-zero entries only at the "
            "first phase of each chain (restart points), concentrated on the current "
            "chain (diagonal blocks dominant)."
        ),
    ),
    (
        "hyper_telegraph",
        "Zone C: negative rho1, high SCV",
        "< -0.3 (negative)",
        "> 3 (high)",
        (
            "Two blocks of hyperexponential states (fast and slow), with D1 strictly "
            "off-diagonal so every arrival switches the block. Fast-block rates are "
            "log-uniformly sampled around rate_ratio x base; slow-block rates around "
            "base. This guarantees rho1 < 0, and within-block rate spread drives "
            "SCV above 3."
        ),
        (
            "D0 is block-diagonal: only the diagonal entries are non-zero within each "
            "block (hyperexponential = independent exponential states, no within-block "
            "transitions). All off-diagonal entries of D0 within each block and all "
            "cross-block entries are zero. D1 is block off-diagonal: upper-right "
            "(fast to slow) and lower-left (slow to fast) are dense; diagonal blocks "
            "of D1 are zero."
        ),
    ),
    (
        "erlang_telegraph",
        "Zone D: negative rho1, low SCV",
        "< 0 (negative)",
        "< 1 (low)",
        (
            "Two Erlang chains with strictly off-diagonal D1: every arrival at the "
            "end of chain A restarts chain B and vice versa. The moderate rate ratio "
            "(1.2-8) keeps SCV well below 1 (Erlang phases reduce variability), while "
            "the alternation creates negative autocorrelation."
        ),
        (
            "D0 is block-diagonal: within each Erlang chain, only the superdiagonal "
            "is non-zero (phase progression), all cross-chain entries are zero. D1 "
            "has exactly two non-zero entries: D1[last_A, first_B] = rate_a and "
            "D1[last_B, first_A] = rate_b. All other D1 entries are zero. This is "
            "the sparsest possible non-trivial D1 for a two-block MAP."
        ),
    ),
    (
        "erlang_rate_spread",
        "Zone E: positive rho1, very high SCV",
        "> 0.5 (positive)",
        "> 5 (very high)",
        (
            "k parallel Erlang chains with biased same-chain restart (like "
            "erlang_modulated), but with one rare slow anchor mode whose rate is "
            "rate_spread times smaller than the fast modes. The rare slow mode "
            "inflates SCV to O(1/weight), while the Erlang memory and biased D1 "
            "restart maintain rho1 > 0.5."
        ),
        (
            "Same sparsity structure as erlang_modulated: D0 is block-diagonal with "
            "Erlang superdiagonals; D1 has non-zero entries only at the last phase of "
            "each chain, pointing to the first phase of each chain. The one structural "
            "difference from erlang_modulated is the magnitude: one chain has rates "
            "1/rate_spread times smaller, making the slow mode rare but visible in "
            "the generated process statistics."
        ),
    ),
    (
        "coxian_alternating",
        "Zone F: negative rho1, moderate SCV",
        "< 0 (negative)",
        "Moderate to high",
        (
            "Two Coxian PH blocks alternating strictly. Each block is a Coxian "
            "chain (forward progression with geometrically decaying phase-exit "
            "probabilities). Only the exits of block A send to block B and vice "
            "versa. The early-exit concentration creates a heavy-right-tail mixture "
            "(high SCV), while the alternation gives negative rho1."
        ),
        (
            "D0 is block-diagonal: within each Coxian chain, the superdiagonal "
            "entries are non-zero (forward progression), and there are no cross-block "
            "entries. D1 is block off-diagonal: each phase i of block A has a "
            "non-zero entry D1[i, first_of_B] (exit to B start); each phase of B "
            "similarly exits to the start of A. Both off-diagonal blocks of D1 are "
            "column vectors (one non-zero column per block)."
        ),
    ),
    (
        "erlang_hyper_telegraph",
        "Boundary: rho1 * SCV approaches -1",
        "Near -1/SCV (theoretical minimum)",
        "2 to 20+ (slides along boundary)",
        (
            "Erlang block A + Hyperexponential block B, strictly alternating D1. "
            "Analytically, as rate_ratio and k_erlang both grow, this structure "
            "achieves rho1 * SCV approaching -1, which is the theoretical minimum "
            "for any non-negative process. By varying hyper_spread (SCV of block B), "
            "the structure slides along the boundary curve rho1 = -1/SCV."
        ),
        (
            "D0 block-diagonal: block A (Erlang) has superdiagonal-only non-zero "
            "entries; block B (hyperexponential) has diagonal-only non-zero entries "
            "(independent exponentials, no within-block transitions). D1 is block "
            "off-diagonal: only the last phase of A can trigger an arrival (row "
            "last_A of D1 is non-zero across all B columns with weights w_j); each "
            "phase of B triggers an arrival that restarts A at its first phase "
            "(D1[B_j, 0] non-zero for all j)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, TITLE, align="R")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, num: int, name: str, zone: str) -> None:
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 240, 255)
        self.set_text_color(20, 40, 100)
        self.cell(0, 8, f"{num}. {name}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 120)
        self.cell(0, 5, f"Zone / Category: {zone}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def stats_row(self, rho1: str, scv: str) -> None:
        col_w = (self.w - self.l_margin - self.r_margin) / 2
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(50, 50, 50)
        self.cell(col_w, 6, f"Autocorrelation (rho1): {rho1}", border="B",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w, 6, f"SCV range: {scv}", border="B",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def body_section(self, label: str, text: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 30, 30)
        self.cell(0, 5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def section_label(self, text: str) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 30, 30)
        self.cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def divider(self) -> None:
        self.set_draw_color(200, 210, 230)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------

def build_pdf() -> None:
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 40, 100)
    pdf.ln(8)
    pdf.multi_cell(0, 10, TITLE, align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(80, 80, 120)
    pdf.multi_cell(0, 7, SUBTITLE, align="C")
    pdf.ln(6)

    pdf.set_draw_color(100, 130, 200)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    intro = (
        "A Markovian Arrival Process (MAP) is a versatile stochastic model for "
        "arrival streams that generalises the Poisson process. It is characterised by "
        "a pair of matrices (D0, D1), both of size n x n, where n is the number of "
        "phases. D0 governs transitions that occur without an arrival, while D1 governs "
        "transitions that coincide with an arrival. The sum Q = D0 + D1 is the generator "
        "of the background continuous-time Markov chain (CTMC).\n\n"
        "Key properties used throughout this document:\n"
        "  - rho1: first-lag autocorrelation of inter-arrival times.\n"
        "  - SCV: squared coefficient of variation of the marginal inter-arrival distribution.\n"
        "  - Renewal process: rho1 = 0 (consecutive inter-arrivals are independent).\n\n"
        "Each section includes a sparsity visualisation: a grid where filled cells "
        "indicate non-zero entries and empty cells indicate zeros. Dark navy marks the "
        "D0 diagonal (always non-zero, always negative), blue marks D0 off-diagonal "
        f"non-zeros, and green marks D1 non-zeros. {len(MAP_TYPES)} MAP families are described."
    )
    pdf.multi_cell(0, 5.5, intro)
    pdf.ln(8)

    # Table of contents
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 40, 100)
    pdf.cell(0, 7, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(180, 180, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    for i, (name, zone, _, _, _, _) in enumerate(MAP_TYPES, 1):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(10, 5, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(60, 5, name, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(100, 100, 140)
        pdf.cell(0, 5, zone, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)

    # One page per MAP type
    for i, (name, zone, rho1, scv, description, structure_text) in enumerate(MAP_TYPES, 1):
        pdf.add_page()
        pdf.chapter_title(i, name, zone)
        pdf.stats_row(rho1, scv)
        pdf.body_section("Description", description)
        pdf.body_section("Matrix Structure", structure_text)

        # Matrix sparsity visualisation
        pdf.section_label("Sparsity Pattern (small example)")
        map_obj = get_example_map(name)
        if map_obj is not None:
            draw_matrix_pair(pdf, map_obj.D0, map_obj.D1)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, "(example unavailable for this type)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.divider()

    pdf.output(str(OUT_PATH))
    print(f"\nPDF saved to: {OUT_PATH}")


if __name__ == "__main__":
    build_pdf()
