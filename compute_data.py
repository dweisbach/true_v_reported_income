# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:38:47 2026
@author: dweisbac

compute_data.py
Executes all simulations on the FULL population and exports lightweight CSV arrays.
Maintains float64 precision for the extreme lognormal tail.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
import tax_model as tm

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
TARGET_MEAN = 65000
BASE_EVASION = 0.10  # Matches empirical aggregate tax gap
WALKTHROUGH_BETA = 0.05
WALKTHROUGH_SIGMA = 1.4
WALKTHROUGH_SCENARIOS = [
    ("Proportional Evasion", 0.0, 0.0),
    ("Progressive, Heterogeneous", WALKTHROUGH_BETA, WALKTHROUGH_SIGMA),
    ("Strong Progressive Heterogeneous", 0.10, 1.4),
    ("Regressive Heterogeneous", -0.05, 1.4),
]
BETA_VALS = np.round(np.arange(-0.10, 0.31, 0.05), 2)
SIGMA_VALS = np.round(np.arange(0.0, 3.1, 0.2), 1)

def gini(x):
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n


# =============================================================================
# MATH ENGINE
# =============================================================================
def target_share_error(guess, shocks, beta, snu, inc_dist, noise_dist, ev_seed, k1_cal):
    """Objective function for the solver, moved outside to fix scoping errors."""
    if inc_dist == 'lognormal':
        mu = np.log(TARGET_MEAN) - (guess**2 / 2)
        y_temp = np.exp(mu + guess * shocks)
    elif inc_dist == 'pareto':
        y_min = TARGET_MEAN * (guess - 1.0) / guess
        y_temp = y_min * (1.0 - shocks)**(-1.0 / guess)
        
    y_rep_temp, _ = tm.apply_evasion(
        y_temp, beta, snu, mode='loglinear', 
        noise_dist=noise_dist, z_type='log_income', 
        base_evasion=BASE_EVASION, seed=ev_seed
    )
    share = np.partition(y_rep_temp, -k1_cal)[-k1_cal:].sum() / y_rep_temp.sum()
    return share - 0.20


def calibrate_theta_baseline(n_agents, inc_dist, noise_dist):
    """
    Calibrates the distribution parameter (theta) once at the neutral baseline 
    (Beta=0, Sigma=0) to achieve a 20% Top 1% share.
    """
    print(f"--- Running One-Time Baseline Calibration for {inc_dist.upper()} ---")
    n_cal = min(n_agents, 2000000) 
    np.random.seed(42)
    
    if inc_dist == 'lognormal':
        shocks = np.random.randn(n_cal)
        box = (0.1, 4.0)
    else:
        shocks = np.random.rand(n_cal)
        box = (1.05, 5.0)
        
    k1 = int(n_cal * 0.01)
    args = (shocks, 0.0, 0.0, inc_dist, noise_dist, 43, k1)
    
    theta_baseline = brentq(target_share_error, box[0], box[1], args=args, xtol=1e-6)
    print(f"  Baseline Theta Found: {theta_baseline:.6f}")
    return theta_baseline


def _build_economy(beta, snu, n_agents, inc_dist, noise_dist, seed=42, ev_seed=43, n_cal_agents=None, fixed_theta=None):
    # 1. CALIBRATION (Uses small proxy, no memory risk here)
    if fixed_theta is not None:
        cal_param = fixed_theta
    else:
        if n_cal_agents is None:
            n_cal_agents = n_agents
            
        np.random.seed(seed)
        
        if inc_dist == 'lognormal':
            cal_shocks = np.random.randn(n_cal_agents)
            search_boxes = [(0.1, 4.0), (0.01, 10.0)]
        else:
            cal_shocks = np.random.rand(n_cal_agents)
            search_boxes = [(1.05, 5.0), (1.001, 25.0)]
            
        k1_cal = int(n_cal_agents * 0.01)
        solver_args = (cal_shocks, beta, snu, inc_dist, noise_dist, ev_seed, k1_cal)
        
        cal_param = None
        for box in search_boxes:
            try:
                cal_param = brentq(target_share_error, box[0], box[1], args=solver_args, xtol=1e-5)
                break
            except ValueError:
                continue

        # Cleanup calibration array immediately
        del cal_shocks 

    # 2. FINAL GENERATION (The Memory-Critical Part)
    np.random.seed(seed)
    
    # Generate shocks directly in the final container to save one full array copy
    if inc_dist == 'lognormal':
        y_true = np.random.randn(n_agents) # Start with Z ~ N(0,1)
        mu_f = np.log(TARGET_MEAN) - (cal_param**2 / 2)
        # Use in-place math: y = exp(mu + sigma * Z)
        y_true *= cal_param
        y_true += mu_f
        np.exp(y_true, out=y_true) # In-place exponentiation
    else:
        y_true = np.random.rand(n_agents) # Start with U ~ U(0,1)
        y_min_f = TARGET_MEAN * (cal_param - 1.0) / cal_param
        # In-place Pareto: y = y_min * (1-U)^(-1/alpha)
        y_true -= 1.0
        y_true *= -1.0
        np.power(y_true, -1.0/cal_param, out=y_true)
        y_true *= y_min_f
        
    # Apply evasion (tm.apply_evasion handles the internal memory for y_rep)
    y_rep, ev_rates = tm.apply_evasion(
        y_true, beta, snu, mode='loglinear', 
        noise_dist=noise_dist, z_type='log_income', 
        base_evasion=BASE_EVASION, seed=ev_seed
    )
    
    return y_true, y_rep, ev_rates, cal_param

# =============================================================================
# 1. CORE GRID COMPUTATIONS
# =============================================================================

def compute_single_cell(beta, snu, n_agents_final, inc_dist='lognormal', noise_dist='beta', fixed_theta=None):
    """Computes standard stats for a single point in the grid using the unified engine."""
    y_true, y_rep, ev_rates, _ = _build_economy(beta, snu, n_agents_final, inc_dist, noise_dist, fixed_theta=fixed_theta)
    
    k1 = int(n_agents_final * 0.01)
    k01 = int(n_agents_final * 0.001)
    
    idx_r = np.argsort(y_rep)
    idx_t = np.argsort(y_true)
    total_true = y_true.sum()
    total_rep = y_rep.sum()
    
    y_true_s = y_true[idx_t]
    y_rep_s = y_rep[idx_r]
    
    s_true_given_rep = y_true[idx_r[-k1:]].sum() / total_true
    
    # --- OPTION A: Dollar-Weighted Evasion of the TRUE Top Percentiles ---
    # 1. Identify the specific individuals in the True Top 1% and 0.1%
    true_top1_idx = idx_t[-k1:]
    true_top01_idx = idx_t[-k01:]
    
    # 2. Calculate their total true dollars and total reported dollars
    true_dollars_1pct = y_true[true_top1_idx].sum()
    rep_dollars_1pct = y_rep[true_top1_idx].sum()
    
    true_dollars_01pct = y_true[true_top01_idx].sum()
    rep_dollars_01pct = y_rep[true_top01_idx].sum()
    
    # 3. Dollar-weighted evasion rate = (True $ - Rep $) / True $
    rate_1pct = (true_dollars_1pct - rep_dollars_1pct) / true_dollars_1pct
    rate_01pct = (true_dollars_01pct - rep_dollars_01pct) / true_dollars_01pct
    
    # 4. Compact heterogeneity stats for the true top 1%
    ev_true_top1 = ev_rates[true_top1_idx]
    p90_evasion_top1 = np.percentile(ev_true_top1, 90)
    frac_true_not_in_rep = 1 - len(set(true_top1_idx) & set(idx_r[-k1:])) / k1
    
    return {
        'Beta': beta, 'Sigma': snu,
        'gap_1pct': (y_rep_s[-k1:].sum() / total_rep) - (y_true_s[-k1:].sum() / total_true),
        'gap_01pct': (y_rep_s[-k01:].sum() / total_rep) - (y_true_s[-k01:].sum() / total_true),
        'rate_1pct': rate_1pct,
        'rate_01pct': rate_01pct,
        'gini_diff': gini(y_rep) - gini(y_true),
        's_true': y_true_s[-k1:].sum() / total_true,
        's_rep': y_rep_s[-k1:].sum() / total_rep,
        's_true_given_rep': s_true_given_rep,
        'agg_gap': (total_true - total_rep) / total_true,
        'p90_evasion_top1': p90_evasion_top1,
        'frac_reranked_top1': frac_true_not_in_rep
    }


def _cell_worker(args):
    """Wrapper to unpack arguments for the parallel workers."""
    return compute_single_cell(*args)


def compute_robustness_grid_fixed(inc_dist, noise_dist, n_agents=10000000):
    """
    The optimized engine for the massive robustness heatmaps using a FIXED calibration.
    Locks theta at the baseline (Beta=0, Sigma=0) for all cells.
    """
    # 1. Calibrate baseline theta once
    theta_star = calibrate_theta_baseline(n_agents, inc_dist, noise_dist)
    
    print(f"\n--- COMPUTING FIXED-THETA GRID: {inc_dist.upper()} / {noise_dist.upper()} ({n_agents:,} AGENTS) ---")
    
    # We pack all 6 arguments into the task list for the worker
    tasks = [(beta, snu, n_agents, inc_dist, noise_dist, theta_star) for beta in BETA_VALS for snu in SIGMA_VALS]
    
    with ProcessPoolExecutor(max_workers=14) as executor:
        results = list(executor.map(_cell_worker, tasks))
        
    filename = f"data_fixed_theta_{inc_dist}_{noise_dist}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"--- SAVED TO {filename} ---")


def compute_robustness_grid(inc_dist, noise_dist, n_agents=10000000):
    """
    The optimized engine for the massive robustness heatmaps.
    Now correctly passes inc_dist and noise_dist to the workers.
    """
    print(f"\n--- COMPUTING {inc_dist.upper()} / {noise_dist.upper()} ({n_agents:,} AGENTS) ---")
    
    # We pack all 5 arguments into the task list for the worker
    tasks = [(beta, snu, n_agents, inc_dist, noise_dist) for beta in BETA_VALS for snu in SIGMA_VALS]
    
    with ProcessPoolExecutor(max_workers=14) as executor:
        results = list(executor.map(_cell_worker, tasks))
        
    filename = f"data_big_{inc_dist}_{noise_dist}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"--- SAVED TO {filename} ---")   
    
def compute_core_grid(n_agents=1000000):
    """The baseline grid for the main text and Table 1."""
    print(f"--- COMPUTING SMALL BASELINE GRID ({n_agents:,} AGENTS) ---")
    results = []
    
    small_beta = [-0.10, -0.05, 0.00, 0.05, 0.10]
    small_sigma = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    
    for beta in small_beta:
        for snu in small_sigma:
            print(f"Computing Baseline Cell: Gamma={beta}, Sigma={snu}")
            # Explicitly request 'normal' noise to match your original heatmaps
            res = compute_single_cell(beta, snu, n_agents, noise_dist='beta')
            results.append(res)
            
    pd.DataFrame(results).to_csv("data_core_grid.csv", index=False)
    print("--- BASELINE GRID COMPLETE ---")


# =============================================================================
# 2. WALKTHROUGH & INTUITION COMPUTATIONS
# =============================================================================

def compute_walkthrough(n_agents_sample=10000000, n_target_pop=330000000, inc_dist='lognormal', noise_dist='beta',
                        scenarios=None):
    """
    Computes scaled walkthrough data and Option A line-chart curves.
    Uses a RAM-safe sample and scales aggregates to the full US population.
    
    Parameters
    ----------
    scenarios : list of (label, beta, sigma_nu) tuples, optional
        Each tuple defines one row of the walkthrough figure.
        Default: uses WALKTHROUGH_SCENARIOS module constant.
    """
    if scenarios is None:
        scenarios = WALKTHROUGH_SCENARIOS
    
    scale_factor = n_target_pop / n_agents_sample 
    
    print(f"--- COMPUTING SCALED WALKTHROUGH ({len(scenarios)} scenarios) ---")
    print(f"Scaling factor: {scale_factor:.1f}x ({n_target_pop:,} target)")

    decomp_list = []   # Share decomposition stats for the walkthrough table
    profile_list = []  # Heterogeneity profile stats

    for idx, (label, beta, sigma_nu) in enumerate(scenarios):
        print(f"\n  Scenario {idx}: {label} (Beta={beta}, Sigma={sigma_nu})")
        
        # --- Build the economy ---
        y_true, y_rep, ev_rates, _ = _build_economy(
            beta, sigma_nu, n_agents_sample, inc_dist, noise_dist
        )
        
        # --- KDE subsample ---
        n_kde = min(1_000_000, n_agents_sample)
        kde_idx = np.random.choice(n_agents_sample, n_kde, replace=False)
        pd.DataFrame({
            'True': y_true[kde_idx],
            'Reported': y_rep[kde_idx]
        }).to_csv(f"data_walkthrough_kde_{idx}.csv", index=False)
        
        # --- Panel A stats ---
        k1 = int(n_agents_sample * 0.01)
        cutoff_true = np.partition(y_true, -k1)[-k1:].min()
        cutoff_rep = np.partition(y_rep, -k1)[-k1:].min()
        total_true_scaled = y_true.sum() * scale_factor
        total_rep_scaled = y_rep.sum() * scale_factor
        tax_gap = (y_true.sum() - y_rep.sum()) / y_true.sum()
        pd.DataFrame([{
            'Label': label, 'Beta': beta, 'Sigma': sigma_nu,
            'Cutoff_True': cutoff_true,
            'Cutoff_Rep': cutoff_rep,
            'TargetMean': TARGET_MEAN,
            'Total_True': total_true_scaled,
            'Total_Rep': total_rep_scaled,
            'Tax_Gap': tax_gap
        }]).to_csv(f"data_walkthrough_panel_stats_{idx}.csv", index=False)
        
        # --- Option A line chart curves ---
        print("  Computing Dollar-Weighted Evasion Curves...")
        idx_t = np.argsort(y_true)
        idx_r = np.argsort(y_rep)
        
        # --- Share decomposition for walkthrough table ---
        total_true = y_true.sum()
        total_rep = y_rep.sum()
        s_true = y_true[idx_t[-k1:]].sum() / total_true
        s_rep = y_rep[idx_r[-k1:]].sum() / total_rep
        s_true_given_rep = y_true[idx_r[-k1:]].sum() / total_true
        s_rep_given_true = y_rep[idx_t[-k1:]].sum() / total_rep
        
        decomp_list.append({
            'Step': label, 'Beta': beta, 'Sigma': sigma_nu,
            's_true': s_true, 's_rep': s_rep,
            's_true_given_rep': s_true_given_rep,
            's_rep_given_true': s_rep_given_true
        })
        
        # --- Heterogeneity profile ---
        ev_true_top1 = ev_rates[idx_t[-k1:]]
        true_top1_y = y_true[idx_t[-k1:]]
        true_top1_r = y_rep[idx_t[-k1:]]
        unreported_top1 = true_top1_y - true_top1_r
        
        # Concentration: share of top-1% unreported $ from top 10% of evaders
        k_top10 = int(k1 * 0.10)
        top_evaders_idx = np.argsort(unreported_top1)[-k_top10:]
        share_from_top10 = unreported_top1[top_evaders_idx].sum() / unreported_top1.sum()
        
        # Mobility: fraction of true top 1% not in reported top 1%
        true_top1_set = set(idx_t[-k1:])
        rep_top1_set = set(idx_r[-k1:])
        frac_true_not_in_rep = 1 - len(true_top1_set & rep_top1_set) / k1
        
        # Rank correlation (subsample for speed)
        from scipy.stats import spearmanr
        n_corr = min(1_000_000, n_agents_sample)
        corr_idx = np.random.choice(n_agents_sample, n_corr, replace=False)
        rho, _ = spearmanr(y_true[corr_idx], y_rep[corr_idx])
        
        profile_list.append({
            'Scenario': label, 'Beta': beta, 'Sigma': sigma_nu,
            'Aggregate_Gap': tax_gap,
            'DW_Evasion_Top1': (true_top1_y.sum() - true_top1_r.sum()) / true_top1_y.sum(),
            'Median_Evasion_Top1': np.median(ev_true_top1),
            'P75_Evasion_Top1': np.percentile(ev_true_top1, 75),
            'P90_Evasion_Top1': np.percentile(ev_true_top1, 90),
            'Frac_Above_25pct': (ev_true_top1 > 0.25).mean(),
            'Frac_Above_50pct': (ev_true_top1 > 0.50).mean(),
            'Share_From_Top10_Evaders': share_from_top10,
            'Frac_True_Top1_Not_In_Rep_Top1': frac_true_not_in_rep,
            'Spearman_Rho': rho
        })
        
        grid_pct = np.logspace(0, -2, 100)
        
        true_sorted = y_true[idx_t]
        rep_sorted = y_rep[idx_r]
        rep_of_true_top = y_rep[idx_t]
        true_of_rep_top = y_true[idx_r]
        
        ts, rs, es_rep, es_true = [], [], [], []
        
        for p in grid_pct:
            k = max(int(n_agents_sample * (p/100)), 1)
            
            t_dollars = true_sorted[-k:].sum() * scale_factor
            r_dollars = rep_sorted[-k:].sum() * scale_factor
            ts.append(t_dollars)
            rs.append(r_dollars)
            
            evaded_by_true_top = (true_sorted[-k:].sum() - rep_of_true_top[-k:].sum())
            es_true.append(evaded_by_true_top / true_sorted[-k:].sum())
            
            true_dollars_of_rep_top = true_of_rep_top[-k:].sum()
            evaded_by_rep_top = true_dollars_of_rep_top - rep_sorted[-k:].sum()
            es_rep.append(evaded_by_rep_top / true_dollars_of_rep_top)
        
        pd.DataFrame({
            'grid_pct': grid_pct, 
            'true_dollars_usd': ts, 
            'rep_dollars_usd': rs, 
            'evasion_rate_true_top': es_true, 
            'evasion_rate_rep_top': es_rep
        }).to_csv(f"data_walkthrough_scaled_lines_{idx}.csv", index=False)
        
        # --- Evasion profile by income bin (for plot_evasion_profiles) ---
        print("  Computing Evasion Profiles by Income Bin...")
        income_bin_edges = np.logspace(np.log10(100), np.log10(3e6), 50)
        bin_centers = np.sqrt(income_bin_edges[:-1] * income_bin_edges[1:])
        
        bins_by_true = np.digitize(y_true, income_bin_edges) - 1
        bins_by_rep  = np.digitize(np.maximum(y_rep, 1), income_bin_edges) - 1
        
        n_bins = len(bin_centers)
        avg_by_true = np.full(n_bins, np.nan)
        avg_by_rep  = np.full(n_bins, np.nan)
        n_obs_true  = np.zeros(n_bins, dtype=int)
        n_obs_rep   = np.zeros(n_bins, dtype=int)
        
        for b in range(n_bins):
            mask_t = (bins_by_true == b)
            mask_r = (bins_by_rep == b)
            n_obs_true[b] = mask_t.sum()
            n_obs_rep[b]  = mask_r.sum()
            if n_obs_true[b] > 100:
                avg_by_true[b] = ev_rates[mask_t].mean()
            if n_obs_rep[b] > 100:
                avg_by_rep[b]  = ev_rates[mask_r].mean()
        
        pd.DataFrame({
            'income_center': bin_centers,
            'avg_evasion_by_true': avg_by_true,
            'avg_evasion_by_rep': avg_by_rep,
            'n_obs_true': n_obs_true,
            'n_obs_rep': n_obs_rep,
        }).to_csv(f"data_walkthrough_evasion_profile_{idx}.csv", index=False)
        
        del y_true, y_rep, ev_rates

    # Save scenario metadata for the plotting script
    pd.DataFrame([
        {'idx': i, 'Label': label, 'Beta': b, 'Sigma': s}
        for i, (label, b, s) in enumerate(scenarios)
    ]).to_csv("data_walkthrough_scenarios.csv", index=False)

    # Save share decomposition table
    pd.DataFrame(decomp_list).to_csv("data_walkthrough_stats.csv", index=False)

    # Save heterogeneity profile table
    pd.DataFrame(profile_list).to_csv("data_heterogeneity_profile.csv", index=False)

    print("\nWalkthrough data saved for all scenarios.")





        # M
def calculate_parameter_intuition(gamma=0.05, sigma_nu=1.4, inc_dist='lognormal', noise_dist='beta', n_agents=5000000):
    """
    Calculates reporting intuition using the unified math engine.
    Now includes the Aggregate Dollar-Weighted Gap and Dollar-Weighted Top 1% Evasion.
    """
    print(f"\n--- Calculating Intuition for Gamma={gamma}, Sigma={sigma_nu} ---")
    print(f"--- Economy: {inc_dist.capitalize()} True Income / {noise_dist.capitalize()} Evasion Noise ---")
    
    # Generate entirely via unified master engine
    y_true, y_rep, ev_rates, _ = _build_economy(gamma, sigma_nu, n_agents, inc_dist, noise_dist)

    k_1pct = int(n_agents * 0.01)
    idx_t = np.argsort(y_true)
    idx_r = np.argsort(y_rep)

    total_true = y_true.sum()
    total_rep = y_rep.sum()

    # 1. Calculate the Aggregate Dollar-Weighted Gap
    aggregate_gap = 1.0 - (total_rep / total_true)

    # 2. Dollar-Weighted Evasion of the True Top 1%
    true_top1_idx = idx_t[-k_1pct:]
    top1_true_dollars = y_true[true_top1_idx].sum()
    top1_rep_dollars = y_rep[true_top1_idx].sum()
    ev_rate_top1_weighted = 1.0 - (top1_rep_dollars / top1_true_dollars)
    
    # 3. Calculate Unweighted Reporting Rates (for reference)
    rep_rates = 1 - ev_rates
    avg_rep_all = rep_rates.mean()
    
    # Calculate Top 1% Shares for the Decomposition
    s_y = y_true[idx_t[-k_1pct:]].sum() / total_true
    s_r = y_rep[idx_r[-k_1pct:]].sum() / total_rep
    s_y_given_r = y_true[idx_r[-k_1pct:]].sum() / total_true

    # The Core Decomposition
    selection_effect = s_y_given_r - s_y
    diff_evasion_effect = s_r - s_y_given_r
    total_gap = s_r - s_y
    
    print("-" * 55)
    print(f"Aggregate Dollar-Weighted Gap:            {aggregate_gap:.1%}  <-- Matches Total Gap Heatmap")
    print(f"Dollar-Weighted Evasion (True Top 1%):    {ev_rate_top1_weighted:.1%}  <-- Matches Top 1% Heatmap")
    print(f"Unweighted Average Reporting Rate:        {avg_rep_all:.1%}")
    
    print("-" * 55)
    print("THE DECOMPOSITION OF THE REPORTED GAP (S_R - S_Y):")
    print(f"  True Share of True Top 1% (S_Y):         {s_y:.4f}")
    print(f"  Reported Share of Reported Top 1% (S_R): {s_r:.4f}")
    print(f"  True Share of Reported Top 1% (S_Y|R):   {s_y_given_r:.4f}")
    print("-" * 55)
    print(f"  1. Selection Effect (S_Y|R - S_Y):       {selection_effect:+.4f}  <-- Pulls reported share DOWN")
    print(f"  2. Differential Evasion (S_R - S_Y|R):   {diff_evasion_effect:+.4f}  <-- Pulls reported share UP")
    print(f"  =======================================================")
    print(f"  Net Reported Gap (S_R - S_Y):            {total_gap:+.4f}")
    print("-" * 55)

    # Memory hygiene
    del y_true, y_rep, ev_rates
# =============================================================================
# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    import time
    start_time = time.time()

    print("--- tax_model.py Execution: Full Production ---")

    # 1. PARAMETER INTUITION (Theoretical Foundation)
    # Explains reporting behavior for Step 3: Gamma=0.05, Sigma=1.4.
    #calculate_parameter_intuition(
    #    gamma=0.05, 
    #    sigma_nu=1.4, 
    #    inc_dist='lognormal', 
    #    noise_dist='beta', 
    #    n_agents=5000000
    #)

    # 2. SCALED WALKTHROUGH (Main Figures)
    # Generates totals for 330M population and Option A line curves.
    compute_walkthrough(n_agents_sample=10000000, n_target_pop=330000000, inc_dist='lognormal', noise_dist='beta')

    # 3. BASELINE GRID (Table 1)
    # A smaller, high-precision run for the main text tables.
    #compute_core_grid(n_agents=5000000)

    # 4. FULL 2x2 ROBUSTNESS GRIDS (Production Heatmaps)
    # Covers the full matrix: (Lognormal/Pareto) x (Beta/Normal).
    # We use 10M agents for each to ensure stability in the extreme tails.
    
    # Quadrant 1: Lognormal Income / Beta Evasion Noise
    #compute_robustness_grid(inc_dist='lognormal', noise_dist='beta', n_agents=10000000)
    
    # Quadrant 2: Lognormal Income / Normal Evasion Noise
    #compute_robustness_grid(inc_dist='lognormal', noise_dist='normal', n_agents=10000000)
    
    # Quadrant 3: Pareto Income / Beta Evasion Noise
    #compute_robustness_grid(inc_dist='pareto', noise_dist='beta', n_agents=10000000)
    
    # Quadrant 4: Pareto Income / Normal Evasion Noise
    #compute_robustness_grid(inc_dist='pareto', noise_dist='normal', n_agents=10000000)
    
    # --- NEW FIXED-THETA RUNS ---
    print("\n--- RUNNING FIXED-THETA ROBUSTNESS GRIDS ---")
    
    # 1. Beta Noise Runs (You already have these CSVs, you can comment them out if you want to save time)
    # compute_robustness_grid_fixed(inc_dist='lognormal', noise_dist='beta', n_agents=10000000)
    # compute_robustness_grid_fixed(inc_dist='pareto', noise_dist='beta', n_agents=10000000)

    # 2. Normal (Log-Linear) Noise Runs (These are the ones missing)
    #compute_robustness_grid_fixed(inc_dist='lognormal', noise_dist='normal', n_agents=10000000)
    #compute_robustness_grid_fixed(inc_dist='pareto', noise_dist='normal', n_agents=10000000)


    elapsed = time.time() - start_time
    print(f"\n--- Total Execution Time: {elapsed/60:.2f} minutes ---")