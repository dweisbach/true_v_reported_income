# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:59:36 2026

@author: dweisbac
"""

# -*- coding: utf-8 -*-
"""
simulate_two_type.py
 
Two-component (wage + pass-through) income model, additive version.
 
    wage income          w  ~ Lognormal(mean_w, sigma_w)   reported at rate (1 - e_w)
    pass-through income  pt ~ Lognormal(mean_p, sigma_p)    reported at rate (1 - e_p)
 
    true income      y = w + pt
    reported income  r = (1 - e_w) * w + (1 - e_p) * pt
 
This is the literal Appendix B structure (y = visible + opaque, evasion applied
to the opaque component). Because the total is the SUM of two lognormals, it is
not itself lognormal; both the true and reported distributions are still well
defined, so all share comparisons are valid.
 
Useful identities (with e_w = 0):
    aggregate pass-through share  = sum(pt) / sum(y) = mean_p / (mean_w + mean_p)
    aggregate tax gap             = e_p * (aggregate pass-through share)
The aggregate share depends only on the ratio of means, so sigma_p is a clean
lever for top-share concentration that leaves the aggregate share untouched.
"""
 
import numpy as np
import pandas as pd
 
 
# ----------------------------------------------------------------------
# Core simulation
# ----------------------------------------------------------------------
def lognormal_mu(mean, sigma):
    """mu of the underlying normal so that Lognormal(mu, sigma) has E[X] = mean."""
    return np.log(mean) - 0.5 * sigma ** 2
 
 
def simulate(N, mean_w, sigma_w, mean_p, sigma_p, e_p, e_w=0.0, seed=None):
    """Draw a population and return (true income, reported income, wage, pass-through)."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(lognormal_mu(mean_w, sigma_w), sigma_w, N)
    pt = rng.lognormal(lognormal_mu(mean_p, sigma_p), sigma_p, N)
    y = w + pt
    r = (1.0 - e_w) * w + (1.0 - e_p) * pt
    return y, r, w, pt
 
 
# ----------------------------------------------------------------------
# Inequality / decomposition metrics
# ----------------------------------------------------------------------
def top_idx(rank_by, z=0.01):
    """Indices of the top z fraction ranked by `rank_by` (unordered)."""
    n_top = max(1, int(round(z * rank_by.size)))
    return np.argpartition(rank_by, -n_top)[-n_top:]
 
 
def gini(x, max_n=20_000_000, seed=0):
    """Gini coefficient. Subsamples for tractability when x is very large."""
    if x.size > max_n:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_n, replace=False)
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * xs) / (n * xs.sum()))
 
 
def compute_metrics(y, r, z=0.01):
    """All the inequality measures and the rank-mismatch / compliance decomposition."""
    n = y.size
    ti = top_idx(y, z)       # true top group
    ri = top_idx(r, z)       # reported top group
 
    Y, R = y.sum(), r.sum()
    s_true = y[ti].sum() / Y                 # true top share
    s_rep = r[ri].sum() / R                  # reported top share
    s_true_given_rep = y[ri].sum() / Y       # S_{Y|R}: true income of reported top group
 
    gap = s_rep - s_true
    rank_mismatch = s_true_given_rep - s_true            # weakly negative
    compliance_composition = s_rep - s_true_given_rep    # selection term
 
    m = y - r                                # misreported amount
    rate_1pct = m[ti].sum() / y[ti].sum()    # dollar-weighted evasion rate, true top
 
    in_true = np.zeros(n, dtype=bool); in_true[ti] = True
    in_rep = np.zeros(n, dtype=bool); in_rep[ri] = True
    frac_reranked = 1.0 - (in_true & in_rep).sum() / in_true.sum()
 
    return {
        "s_true": s_true,
        "s_rep": s_rep,
        "s_true_given_rep": s_true_given_rep,
        "gap_1pct": gap,
        "rank_mismatch": rank_mismatch,
        "compliance_composition": compliance_composition,
        "rate_1pct": rate_1pct,
        "gini_true": gini(y),
        "gini_rep": gini(r),
        "gini_diff": gini(r) - gini(y),
        "agg_gap": (Y - R) / Y,
        "frac_reranked": frac_reranked,
    }
 
 
# ----------------------------------------------------------------------
# Calibration: choose sigma_p to hit a target reported top share
# ----------------------------------------------------------------------
def calibrate_sigma_p(N, mean_w, sigma_w, mean_p, e_p, target=0.20, z=0.01,
                      e_w=0.0, lo=0.5, hi=4.0, tol=1e-4, max_iter=50, seed=0):
    """
    Bisection on sigma_p so the reported top-z share equals `target`.
    The same seed is reused across evaluations so the objective is smooth in sigma_p.
    """
    def rep_share(sp):
        y, r, _, _ = simulate(N, mean_w, sigma_w, mean_p, sp, e_p, e_w, seed=seed)
        return r[top_idx(r, z)].sum() / r.sum()
 
    f_lo = rep_share(lo) - target
    f_hi = rep_share(hi) - target
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Target {target:.3f} not bracketed on sigma_p in [{lo}, {hi}]: "
            f"reported shares run {f_lo + target:.3f} to {f_hi + target:.3f}. "
            f"Widen the bracket or adjust mean_p / sigma_w."
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = rep_share(mid) - target
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
 
 
# ----------------------------------------------------------------------
# Optional: sweep over aggregate pass-through share and evasion rate
# ----------------------------------------------------------------------
def run_grid(N, mean_w, sigma_w, pt_share_vals, e_p_vals, outfile,
             target=0.20, z=0.01, seed=0):
    """For each (pass-through share, e_p), recalibrate sigma_p and record metrics."""
    rows = []
    for s in pt_share_vals:
        mean_p = s / (1.0 - s) * mean_w      # gives aggregate share == s exactly
        for e_p in e_p_vals:
            sp = calibrate_sigma_p(N, mean_w, sigma_w, mean_p, e_p,
                                   target=target, z=z, seed=seed)
            y, r, w, pt = simulate(N, mean_w, sigma_w, mean_p, sp, e_p, seed=seed)
            M = compute_metrics(y, r, z)
            M.update({
                "pt_share": s,
                "e_p": e_p,
                "mean_p": mean_p,
                "sigma_p": sp,
                "pt_share_realized": pt.sum() / y.sum(),
            })
            rows.append(M)
            print(f"  pt_share={s:.2f}  e_p={e_p:.2f}  ->  "
                  f"sigma_p={sp:.3f}  gap={M['gap_1pct']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"Saved grid to {outfile}")
    return df
 
 
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use a smaller N for development; raise toward 330e6 for a headline run.
    N = 5_000_000
 
    MEAN_W, SIGMA_W = 25_000, 0.7    # wage: modest dispersion, no evasion
    MEAN_P = 40_000                  # pass-through mean -> aggregate share = 15k/65k = 0.231
    E_P, E_W = 0.35, 0.0             # evade 30% of pass-through, nothing on wage
 
    print("Calibrating sigma_p to a 20% reported top-1% share ...")
    sigma_p = calibrate_sigma_p(N, MEAN_W, SIGMA_W, MEAN_P, E_P,
                                target=0.20, e_w=E_W, seed=0)
    print(f"  sigma_p = {sigma_p:.4f}\n")
 
    y, r, w, pt = simulate(N, MEAN_W, SIGMA_W, MEAN_P, sigma_p, E_P, E_W, seed=0)
    M = compute_metrics(y, r)
 
    pt_share = pt.sum() / y.sum()
    print("=== Baseline economy ===")
    print(f"Mean true income             : ${y.mean():,.0f}")
    print(f"Aggregate pass-through share : {pt_share:6.3f}")
    print(f"Aggregate gap                : {M['agg_gap']:6.3f}   "
          f"(identity check e_p*share = {E_P * pt_share:.3f})")
    print(f"True top 1% share            : {M['s_true']:6.4f}")
    print(f"Reported top 1% share        : {M['s_rep']:6.4f}")
    print(f"Reported income gap          : {M['gap_1pct']:+6.4f}")
    print(f"  Rank-mismatch              : {M['rank_mismatch']:+6.4f}")
    print(f"  Compliance-composition     : {M['compliance_composition']:+6.4f}")
    print(f"Avg evasion, true top 1%     : {M['rate_1pct']:6.4f}")
    print(f"Frac true top 1% reranked out: {M['frac_reranked']:6.3f}")
    print(f"Gini gap (reported - true)   : {M['gini_diff']:+6.4f}")
 
    # --- Optional sweep (set to True to run) ---
    RUN_GRID = False
    if RUN_GRID:
        print("\nRunning grid ...")
        run_grid(
            N=2_000_000,
            mean_w=MEAN_W, sigma_w=SIGMA_W,
            pt_share_vals=[0.10, 0.15, 0.20, 0.25],
            e_p_vals=[0.20, 0.30, 0.40],
            outfile="data_two_type_grid.csv",
        )