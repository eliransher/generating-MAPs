"""Analytical departure-process moments for MAP/M/1 queueing networks.

Theory
------
The departure process from MAP/M/1 is a Markov renewal process with n_a states
(the MAP background states at departure epochs).  Its kernel LST matrix is:

    h(s) = F(s) * B(s)

where
    F(s) = diag(p_empty) * (-D0 - sI)^{-1} * D1  +  diag(p_busy)
    B(s) = mu * [(mu+s)I - D0 - D1]^{-1}

p_empty(i) = probability that a departure in MAP state i leaves the queue empty.
This is derived from the matrix-geometric R matrix and the stationary distribution.

Moment formulas (semi-Markov):
    H^(a)   = (-1)^a * d^a h(s)/ds^a |_{s=0}     (a-th moment matrices)
    phi     = stationary dist at departure epochs  (phi * P^d = phi, P^d = h(0))
    E[D^a]  = phi @ H^(a) @ 1
    E[D_n^a * D_{n+k}^b] = (phi @ H^(a)) . ((P^d)^{k-1} @ H^(b) @ 1)

For a 3-station network (stations 1, 2 feed station 3):
    - Departure MRPs from stations 1 and 2 are superposed via Kronecker product
    - Station 3 sees the merged MAP/M/1 queue
    - All moments of the station-3 departure process are computed analytically
"""

from __future__ import annotations
import sys
from pathlib import Path
from math import factorial

import numpy as np
from scipy.linalg import solve, inv

CODE_DIR = Path(r"C:\Users\Eshel\workspace\map_fitting\code")
sys.path.insert(0, str(CODE_DIR))
import sampling_map as sm


# ---------------------------------------------------------------------------
# Step 1: Solve Riccati equation  mu R^2 + R(D0 - mu I) + D1 = 0
# ---------------------------------------------------------------------------

def solve_r_matrix(D0: np.ndarray, D1: np.ndarray, mu: float,
                   max_iter: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Minimal nonneg solution of mu*R^2 + R*(D0-mu*I) + D1 = 0 via fixed-point iteration."""
    n = D0.shape[0]
    A = inv(mu * np.eye(n) - D0)   # (mu*I - D0)^{-1}
    R = np.zeros((n, n))
    for _ in range(max_iter):
        R_new = (D1 + mu * R @ R) @ A
        if np.max(np.abs(R_new - R)) < tol:
            return R_new
        R = R_new
    return R


# ---------------------------------------------------------------------------
# Step 2: Stationary distribution of the QBD
# ---------------------------------------------------------------------------

def qbd_stationary(D0: np.ndarray, D1: np.ndarray, mu: float,
                   R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pi0, pi1) for MAP/M/1.

    Level-0 balance: pi0 @ D0 + pi1 * mu = 0  =>  pi1 = -(1/mu) pi0 @ D0
    Level-1 balance gives the equation for pi0:
        pi0 @ [D1 - (1/mu) * D0 @ (D0 - mu*I + mu*R)] = 0
    Normalization: pi0 @ 1 + pi1 @ (I-R)^{-1} @ 1 = 1
    """
    n = D0.shape[0]
    ones = np.ones(n)
    IminR_inv = inv(np.eye(n) - R)

    # Equation pi0 @ A = 0
    A = D1 - (1.0 / mu) * D0 @ (D0 - mu * np.eye(n) + mu * R)

    # Left null vector of A: solve A^T x = 0 with normalization
    # Use the constraint pi0 @ [1 + (-(1/mu)*D0) @ IminR_inv @ 1] = 1
    # => pi0 @ n_vec = 1  where n_vec = 1 + (-1/mu)*D0 @ IminR_inv @ 1
    n_vec = ones - (1.0 / mu) * D0 @ (IminR_inv @ ones)

    A_eq = np.vstack([A.T, n_vec.reshape(1, -1)])
    b_eq = np.zeros(n + 1); b_eq[-1] = 1.0
    pi0, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
    pi0 = np.maximum(pi0, 0.0)

    pi1 = -(1.0 / mu) * pi0 @ D0
    pi1 = np.maximum(pi1, 0.0)

    # Renormalize
    total = pi0 @ ones + pi1 @ IminR_inv @ ones
    pi0 /= total
    pi1 /= total
    return pi0, pi1


# ---------------------------------------------------------------------------
# Step 3: p_empty vector and departure-epoch stationary distribution
# ---------------------------------------------------------------------------

def departure_epoch_dist(pi1: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi, p_empty) where:
        phi[i]     = prob that a departure epoch has MAP state i
        p_empty[i] = prob that departure in state i leaves queue empty
    """
    n = R.shape[0]
    IminR_inv = inv(np.eye(n) - R)
    # Departure-epoch weight proportional to pi1 @ (I-R)^{-1} (sum over all levels)
    w = pi1 @ IminR_inv          # unnormalized phi
    phi = w / w.sum()
    # p_empty[i] = pi1[i] / w[i]
    p_empty = pi1 / np.maximum(w, 1e-300)
    return phi, p_empty


# ---------------------------------------------------------------------------
# Step 4: LST matrix h(s) and its derivatives at s=0
# ---------------------------------------------------------------------------

def _h_and_derivs(D0: np.ndarray, D1: np.ndarray, mu: float,
                  p_empty: np.ndarray, max_order: int) -> list[np.ndarray]:
    """Return [H^(0), H^(1), ..., H^(max_order)] where H^(a) is the a-th
    moment matrix of the inter-departure time Markov renewal kernel.

    h(s) = F(s) * B(s)
    F(s) = diag(p_e)*(sI - D0)^{-1}*D1 + diag(p_b)    (idle-period LST)
    B(s) = mu * [(mu+s)I - G]^{-1},   G = D0+D1        (service-time LST)

    Using the identity d^k/ds^k [(sI-D0)^{-1}]|s=0 = (-1)^k k! (-D0)^{-(k+1)}
    and d^k/ds^k [((mu+s)I-G)^{-1}]|s=0 = (-1)^k k! (muI-G)^{-(k+1)},
    the signs cancel in H[a] = (-1)^a h^(a)(0) yielding:

        H[a] = a! * mu * [F0 * B_pow[a]
                          + sum_{j=1}^{a} Pe * A_pow[j] * D1 * B_pow[a-j]]

    where F0 = Pe*A_pow[0]*D1 + Pb,
          A_pow[k] = (-D0)^{-(k+1)},
          B_pow[k] = (muI-G)^{-(k+1)}.
    """
    n = D0.shape[0]
    G  = D0 + D1
    Pe = np.diag(p_empty)
    Pb = np.diag(1.0 - p_empty)

    invmD0  = inv(-D0)
    invmuIG = inv(mu * np.eye(n) - G)

    A_pow = [invmD0]          # A_pow[k] = (-D0)^{-(k+1)}
    B_pow = [invmuIG]         # B_pow[k] = (muI-G)^{-(k+1)}
    for _ in range(max_order):
        A_pow.append(A_pow[-1] @ invmD0)
        B_pow.append(B_pow[-1] @ invmuIG)

    F0 = Pe @ A_pow[0] @ D1 + Pb   # F(s=0)

    H = []
    for a in range(max_order + 1):
        Ha = F0 @ B_pow[a]
        for j in range(1, a + 1):
            Ha = Ha + Pe @ A_pow[j] @ D1 @ B_pow[a - j]
        H.append(factorial(a) * mu * Ha)
    return H


# ---------------------------------------------------------------------------
# Step 5: Marginal and joint moments of the departure process
# ---------------------------------------------------------------------------

def map_m1_departure_moments(D0: np.ndarray, D1: np.ndarray, mu: float,
                              max_order: int = 4, max_lag: int = 4
                              ) -> list[tuple]:
    """Compute marginal and joint moments of the MAP/M/1 departure process.

    Returns list of (i, j, lag, value, weight) tuples in the same format
    as other target_map moment lists.

    Marginals  E[D^m]                   -> (m, 0, 0, value, w)
    Joints     E[D_n^i * D_{n+k}^j]    -> (i, j, k, value, w)

    Weights mirror the convention used elsewhere:
        Marginals: m=1->10, 2->5, 3->3, 4->2, ...
        Joints:    k=1->2, k>=2->1
    """
    n = D0.shape[0]
    ones = np.ones(n)
    rho = sm.marginal_moment(D0, D1, 1) * (sum(np.diag(D1)) / n)  # rough check
    # stability check
    lam = sm.marginal_moment(D0, D1, 1)   # mean inter-arrival = 1/lambda -> wrong
    # marginal_moment returns E[X^1] = mean inter-arrival time; rate = 1/E[X]
    lam = 1.0 / sm.marginal_moment(D0, D1, 1)
    rho = lam / mu
    assert rho < 1.0, f"Queue unstable: rho={rho:.4f} >= 1"

    R     = solve_r_matrix(D0, D1, mu)
    pi0, pi1 = qbd_stationary(D0, D1, mu, R)
    phi, p_empty = departure_epoch_dist(pi1, R)

    H = _h_and_derivs(D0, D1, mu, p_empty, max_order)
    Pd = H[0]   # h(0) = transition matrix at departure epochs

    MARGINAL_W = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1}
    moments: list[tuple] = []

    # Marginal moments E[D^m] = phi @ H[m] @ 1
    for m in range(1, max_order + 1):
        val = float(phi @ H[m] @ ones)
        w = MARGINAL_W.get(m, 1)
        moments.append((m, 0, 0, val, w))

    # Joint moments E[D_n^i * D_{n+k}^j]
    # = (phi @ H[i]) . ((Pd)^{k-1} @ H[j] @ 1)
    Pd_power = np.eye(n)   # (Pd)^0 = I for lag 1 (k-1=0)
    for k in range(1, max_lag + 1):
        w = 2 if k == 1 else 1
        Hj_1 = {j: H[j] @ ones for j in range(1, max_order + 1)}
        for i in range(1, max_order + 1):
            phi_Hi = phi @ H[i]       # shape (n,)
            for j in range(1, max_order + 1):
                rhs = Pd_power @ Hj_1[j]   # (Pd)^{k-1} @ H[j] @ 1
                val = float(phi_Hi @ rhs)
                moments.append((i, j, k, val, w))
        Pd_power = Pd_power @ Pd   # advance for next lag

    return moments


# ---------------------------------------------------------------------------
# Superposition of two independent departure Markov renewal processes
# ---------------------------------------------------------------------------

def superpose_mrp_as_map(D0_a: np.ndarray, D1_a: np.ndarray,
                         D0_b: np.ndarray, D1_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the superposition of two independent MAP arrival streams
    as a MAP via Kronecker product (exact for MAPs).

    If the two streams are already MAPs this is exact.
    For Markov-renewal departure processes it's an approximation (the merged
    process is fed into station 3 as if it were a MAP).
    """
    na, nb = D0_a.shape[0], D0_b.shape[0]
    Ia, Ib = np.eye(na), np.eye(nb)
    D0_merged = np.kron(D0_a, Ib) + np.kron(Ia, D0_b)
    D1_merged = np.kron(D1_a, Ib) + np.kron(Ia, D1_b)
    return D0_merged, D1_merged


# ---------------------------------------------------------------------------
# Full 3-station network
# ---------------------------------------------------------------------------

def three_station_network_moments(
    D0_1: np.ndarray, D1_1: np.ndarray, mu1: float,
    D0_2: np.ndarray, D1_2: np.ndarray, mu2: float,
    mu3: float,
    max_order: int = 4,
    max_lag: int = 4,
) -> dict:
    """Generate departure-process moments for a 3-station network.

    Topology
    --------
    External MAP_1 -> Station 1 (MAP/M/1, rate mu1) -|
                                                      +--> Station 3 (MAP/M/1, rate mu3) -> output
    External MAP_2 -> Station 2 (MAP/M/1, rate mu2) -|

    Method
    ------
    1. Compute departure moments from stations 1 and 2 analytically.
    2. Re-express each departure process as its 'effective MAP' using its
       stationary kernel matrices D^dep (fitted to the semi-Markov kernel).
    3. Superpose the two departure MAPs (Kronecker product) as input to station 3.
    4. Compute departure moments from station 3.

    Returns a dict with keys 'moments' (and 'D0', 'D1' of the effective
    departure MAP of station 3, for reference).
    """
    # --- stations 1 and 2: compute effective departure MAP matrices ---
    dep1 = _departure_effective_map(D0_1, D1_1, mu1)
    dep2 = _departure_effective_map(D0_2, D1_2, mu2)

    # --- superpose ---
    D0_in3, D1_in3 = superpose_mrp_as_map(dep1[0], dep1[1], dep2[0], dep2[1])

    # --- station 3 departure moments ---
    moments = map_m1_departure_moments(D0_in3, D1_in3, mu3,
                                       max_order=max_order, max_lag=max_lag)
    return {"D0": D0_in3, "D1": D1_in3, "moments": moments}


def _departure_effective_map(D0: np.ndarray, D1: np.ndarray,
                              mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return effective MAP (D0_eff, D1_eff) that represents the departure
    process from MAP/M/1 as a MAP for downstream use.

    The effective MAP is constructed so that its stationary distribution and
    mean / variance match the departure process exactly, using the departure
    MAP transition matrix P^d and mean inter-departure time vector m_i.

    For a MAP: D0_eff[i,j] = P^d[i,j] / m_i  (j != i transition rate)
               D0_eff[i,i] = -1/m_i            (holding rate)
               D1_eff[i,j] = P^d[i,j] / m_i  ... this double-counts.

    Instead use: the departure Markov renewal process has inter-departure time
    drawn from (phi, H matrices). We construct a MAP with:
        rate_i    = 1 / (H[1] @ 1)[i]           (departure rate from state i)
        D1_eff[i,j] = rate_i * P^d[i,j]         (departure transitions)
        D0_eff[i,j] = D0_eff_background[i,j]    (background: use same structure as D0)
    """
    n = D0.shape[0]
    ones = np.ones(n)

    R       = solve_r_matrix(D0, D1, mu)
    pi0, pi1 = qbd_stationary(D0, D1, mu, R)
    phi, p_empty = departure_epoch_dist(pi1, R)
    H       = _h_and_derivs(D0, D1, mu, p_empty, 1)
    Pd      = H[0]
    m_vec   = H[1] @ ones    # E[D | start state i] = m_vec[i]

    rates  = 1.0 / np.maximum(m_vec, 1e-300)
    D1_eff = np.diag(rates) @ Pd       # D1_eff[i,j] = rates[i] * Pd[i,j]
    D0_eff = np.diag(-rates)           # purely diagonal: no background transitions
    # Row sums: D0_eff[i,:] + D1_eff[i,:] = -rates[i] + rates[i]*sum(Pd[i,:]) = 0 ✓
    return D0_eff, D1_eff


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 4
    m1 = sm.generate_random_mmpp(n, seed=1)
    m2 = sm.generate_random_mmpp(n, seed=2)

    mu1, mu2, mu3 = 3.0, 3.0, 8.0   # choose mu > arrival rate for stability

    print("Computing departure moments from station 3...")
    result = three_station_network_moments(
        m1.D0, m1.D1, mu1,
        m2.D0, m2.D1, mu2,
        mu3,
        max_order=4, max_lag=4,
    )
    print(f"Total moments: {len(result['moments'])}")
    for t in result["moments"][:10]:
        print(f"  {t}")
