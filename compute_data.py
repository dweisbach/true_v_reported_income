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
        'agg_gap': (total_true - total_rep) / total_true
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
def compute_walkthrough(n_agents_sample=10000000, n_target_pop=330000000, inc_dist='lognormal', noise_dist='beta'):
    """
    Computes scaled walkthrough data and Option A line-chart curves.
    Generates KDE data and Cutoff statistics required for Panel A plotting.
    Uses a RAM-safe sample and scales aggregates to the full US population.
    """
    scale_factor = n_target_pop / n_agents_sample 
    
    print(f"--- COMPUTING SCALED WALKTHROUGH (Option A Consistent) ---")
    print(f"Scaling factor: {scale_factor:.1f}x ({n_target_pop:,} target)")

    BETA, SIGMA_NU = 0.0, 0.0
    steps = [
        ("1. Baseline", 0.00, 0.0),
        ("2. Add Progressivity", BETA, 0.0),
        ("3. Add Heterogeneity", BETA, SIGMA_NU)
    ]

    stats_list = []
    final_y_true, final_y_rep, final_idx_t, final_idx_r = None, None, None, None

    for step_name, b, snu in steps:
        y_true, y_rep, _, _ = _build_economy(
            b, snu, n_agents_sample, inc_dist, noise_dist
        )
        
        total_true = y_true.sum()
        total_rep = y_rep.sum()
        
        # Calculate cutoffs for the 1% 
        k1 = int(n_agents_sample * 0.01)
        idx_t_temp = np.argsort(y_true)
        idx_r_temp = np.argsort(y_rep)
        
        cutoff_true = y_true[idx_t_temp[-k1]]
        cutoff_rep = y_rep[idx_r_temp[-k1]]

        stats_list.append({
            'Step': step_name,
            'Beta': b,
            'Sigma': snu,
            'Total_True_USD': total_true * scale_factor,
            'Total_Reported_USD': total_rep * scale_factor,
            'Tax_Gap': (total_true - total_rep) / total_true,
            'Cutoff_True': cutoff_true,
            'Cutoff_Rep': cutoff_rep,
            'TargetMean': TARGET_MEAN
        })

        if step_name == "3. Add Heterogeneity":
            final_y_true = y_true
            final_y_rep = y_rep
            final_idx_t = idx_t_temp
            final_idx_r = idx_r_temp
        else:
            del y_true, y_rep 

    pd.DataFrame(stats_list).to_csv("data_walkthrough_scaled_stats.csv", index=False)

    # --- PANEL A KDE EXPORT ---
    print("Exporting KDE Sample for Panel A...")
    # Sample 500k to prevent massive CSV files while maintaining smooth KDEs
    np.random.seed(42)
    sample_indices = np.random.choice(n_agents_sample, min(500000, n_agents_sample), replace=False)
    pd.DataFrame({
        'True': final_y_true[sample_indices],
        'Reported': final_y_rep[sample_indices]
    }).to_csv("data_walkthrough_kde.csv", index=False)

    # --- PANEL B LINE CHART LOGIC ---
    print("Computing Dollar-Weighted Evasion Curves...")
    grid_pct = np.logspace(0, -2, 100) 
    
    true_sorted = final_y_true[final_idx_t]
    rep_sorted = final_y_rep[final_idx_r]
    rep_of_true_top = final_y_rep[final_idx_t]
    true_of_rep_top = final_y_true[final_idx_r]

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
    }).to_csv("data_walkthrough_scaled_lines.csv", index=False)

    print("Walkthrough and Line Data saved.")

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