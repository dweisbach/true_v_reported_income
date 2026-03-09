# -*- coding: utf-8 -*-
"""
compute_data.py
Executes all simulations on the FULL population and exports lightweight CSV arrays.
Maintains float64 precision for the extreme lognormal tail.
"""

import numpy as np
import pandas as pd
import tax_model as tm
import warnings

warnings.filterwarnings("ignore")

BETA_VALS = np.round(np.arange(-0.10, 0.11, 0.05), 2)
SIGMA_VALS = np.round(np.arange(0.0, 1.8, 0.2), 1)

def gini(x):
    sorted_x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

# =============================================================================
# 1. CORE GRID COMPUTATIONS
# =============================================================================
def compute_core_grid(n_agents=2000000):
    print("--- COMPUTING CORE GRID DATA ---")
    results = []
    
    for beta in BETA_VALS:
        for snu in SIGMA_VALS:
            print(f"  Simulating Core Grid: Beta={beta:.2f}, Sigma={snu:.1f}...", end="\r")
            df, _ = tm.get_calibrated_scenario('lognormal', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=n_agents)
            
            k1, k01 = int(n_agents*0.01), int(n_agents*0.001)
            top1, top01 = df.nlargest(k1, 'True'), df.nlargest(k01, 'True')
            top1_rep = df.nlargest(k1, 'Reported')
            top01_rep = df.nlargest(k01, 'Reported')
            
            t1_share = top1['True'].sum() / df['True'].sum()
            r1_share = top1_rep['Reported'].sum() / df['Reported'].sum()
            t01_share = top01['True'].sum() / df['True'].sum()
            r01_share = top01_rep['Reported'].sum() / df['Reported'].sum()
            
            results.append({
                'Beta': beta, 'Sigma': snu,
                'rate_1pct': (top1['True'] - top1['Reported']).sum() / top1['True'].sum(),
                'rate_01pct': (top01['True'] - top01['Reported']).sum() / top01['True'].sum(),
                'agg_gap': (df['True'].sum() - df['Reported'].sum()) / df['True'].sum(),
                'gini_diff': gini(df['True'].values) - gini(df['Reported'].values),
                'gap_1pct': t1_share - r1_share, 
                'gap_01pct': t01_share - r01_share,
                's_true': t1_share, 's_rep': r1_share, 
                's_rep_given_true': df.loc[top1.index, 'Reported'].sum() / df['Reported'].sum()
            })
            
            # --- EXACT MATH ON FULL POPULATION FOR KEY SCENARIOS ---
            key_scenarios = [(0.10, 0.4), (0.10, 1.4), (-0.05, 0.4), (-0.05, 1.4)]
            if (np.round(beta, 2), np.round(snu, 1)) in key_scenarios:
                
                # 1. Exact Share Lines
                start_log, end_log = 1.0, 3.0
                x_log = np.linspace(start_log, end_log, 50)
                q_grid = 1 - 10**(-x_log)
                
                tot_r, tot_t = df['Reported'].sum(), df['True'].sum()
                sort_r = df.sort_values('Reported', ascending=False)['Reported'].values
                sort_t = df.sort_values('True', ascending=False)['True'].values
                
                rep_c, true_c = [], []
                for q in q_grid:
                    k = max(int(n_agents * (1 - q)), 1)
                    rep_c.append(sort_r[:k].sum() / tot_r)
                    true_c.append(sort_t[:k].sum() / tot_t)
                pd.DataFrame({'x_log': x_log, 'rep_c': rep_c, 'true_c': true_c}).to_csv(f"data_core_lines_b{beta:.2f}_s{snu:.1f}.csv", index=False)
                
                # 2. Exact Bins & Evasion Profiles
                bins_log = np.linspace(start_log, end_log, 21)
                bins_q = 1 - 10**(-bins_log)
                df['Rank_True'] = df['True'].rank(pct=True)
                df['Bin'] = pd.cut(df['Rank_True'], bins=bins_q)
                ev_profile = df.groupby('Bin', observed=False)['EvasionRate'].mean().values
                bin_centers_log = (bins_log[:-1] + bins_log[1:]) / 2
                pd.DataFrame({'bin_centers_log': bin_centers_log, 'ev_profile': ev_profile}).to_csv(f"data_core_bins_b{beta:.2f}_s{snu:.1f}.csv", index=False)
                
                lower, upper = df['True'].quantile(0.001), df['True'].quantile(0.999)
                bins = np.logspace(np.log10(lower), np.log10(upper), 30)
                centers = (bins[:-1] + bins[1:]) / 2
                ev_true = df.groupby(pd.cut(df['True'], bins), observed=False)['EvasionRate'].mean().values
                ev_rep = df.groupby(pd.cut(df['Reported'], bins), observed=False)['EvasionRate'].mean().values
                pd.DataFrame({'centers': centers, 'ev_true': ev_true, 'ev_rep': ev_rep}).to_csv(f"data_core_evinc_b{beta:.2f}_s{snu:.1f}.csv", index=False)
                
                # 3. Visual KDE Sample (Safe to sample down for visual curves only)
                df.sample(20000, random_state=42).to_csv(f"data_core_kde_b{beta:.2f}_s{snu:.1f}.csv", index=False)
                
    pd.DataFrame(results).to_csv("data_core_grid.csv", index=False)
    print("\nCore grid data saved.")

# =============================================================================
# 2. WALKTHROUGH CLEAN
# =============================================================================
def compute_walkthrough(n_agents=5000000):
    print("--- COMPUTING WALKTHROUGH DATA ---")
    TARGET_MEAN, BETA, SIGMA_NU = 65000, 0.05, 1.4
    
    low, high, cal_sigma = 0.1, 4.0, 1.0
    for _ in range(50):
        guess = (low + high) / 2
        np.random.seed(1234)
        y_temp = np.random.lognormal(np.log(TARGET_MEAN) - (guess**2 / 2), guess, 100000)
        y_rep_temp, _ = tm.apply_evasion(y_temp, BETA, SIGMA_NU, mode='loglinear', z_type='log_income')
        share = np.sort(y_rep_temp)[-1000:].sum() / y_rep_temp.sum()
        if abs(share - 0.20) < 0.00005: cal_sigma = guess; break
        elif share < 0.20: low = guess
        else: high = guess
            
    np.random.seed(42)
    y_true = np.random.lognormal(np.log(TARGET_MEAN) - (cal_sigma**2 / 2), cal_sigma, n_agents)
    y_rep, ev_rates = tm.apply_evasion(y_true, BETA, SIGMA_NU, mode='loglinear', z_type='log_income', seed=43)
    
    # 1. Exact Array Math on Full Population
    grid_pct = np.logspace(0, -2, 100) 
    tsort, rsort = np.sort(y_true), np.sort(y_rep)
    idx_r, idx_t = np.argsort(y_rep), np.argsort(y_true)
    ev_by_rep, ev_by_true = ev_rates[idx_r], ev_rates[idx_t]
    
    ts, rs, es_rep, es_true = [], [], [], []
    for p in grid_pct:
        k = max(int(n_agents * (p/100)), 1)
        ts.append(tsort[-k:].sum() / tsort.sum())
        rs.append(rsort[-k:].sum() / rsort.sum())
        es_rep.append(ev_by_rep[-k:].mean())   
        es_true.append(ev_by_true[-k:].mean()) 

    pd.DataFrame({'grid_pct': grid_pct, 'ts': ts, 'rs': rs, 'es_rep': es_rep, 'es_true': es_true}).to_csv("data_walkthrough_lines.csv", index=False)
    pd.DataFrame([{'TargetMean': TARGET_MEAN, 'CalSigma': cal_sigma, 'Cutoff_True': np.percentile(y_true, 99), 'Cutoff_Rep': np.percentile(y_rep, 99)}]).to_csv("data_walkthrough_stats.csv", index=False)
    
    # 2. KDE Sample
    pd.DataFrame({'True': y_true, 'Reported': y_rep, 'EvasionRate': ev_rates}).sample(20000, random_state=42).to_csv("data_walkthrough_kde.csv", index=False)
    print("Walkthrough data saved.")

# =============================================================================
# 3. ROBUSTNESS (ADDITIVE & PARETO)
# =============================================================================
def compute_robustness_grid(n_agents=2000000):
    print("--- COMPUTING PARETO/ADDITIVE ROBUSTNESS ---")
    results = []
    for beta in BETA_VALS:
        for snu in SIGMA_VALS:
            print(f"  Simulating Robustness: Beta={beta:.2f}, Sigma={snu:.1f}...", end="\r")
            df_la, _ = tm.get_calibrated_scenario('lognormal', beta=beta, sigma_nu=snu, mode='additive', n_agents=n_agents)
            k1, k01 = int(n_agents*0.01), int(n_agents*0.001)
            la_1 = (df_la.nlargest(k1, 'True')['True'].sum()/df_la['True'].sum()) - (df_la.nlargest(k1, 'Reported')['Reported'].sum()/df_la['Reported'].sum())
            la_01 = (df_la.nlargest(k01, 'True')['True'].sum()/df_la['True'].sum()) - (df_la.nlargest(k01, 'Reported')['Reported'].sum()/df_la['Reported'].sum())

            df_pm, am = tm.get_calibrated_scenario('pareto', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=n_agents)
            pm_1 = (df_pm.nlargest(k1, 'True')['True'].sum()/df_pm['True'].sum()) - (df_pm.nlargest(k1, 'Reported')['Reported'].sum()/df_pm['Reported'].sum())
                                
            df_pa, aa = tm.get_calibrated_scenario('pareto', beta=beta, sigma_nu=snu, mode='additive', n_agents=n_agents)
            pa_1 = (df_pa.nlargest(k1, 'True')['True'].sum()/df_pa['True'].sum()) - (df_pa.nlargest(k1, 'Reported')['Reported'].sum()/df_pa['Reported'].sum())

            results.append({
                'Beta': beta, 'Sigma': snu,
                'log_add_1pct': la_1, 'log_add_01pct': la_01,
                'par_mult_1pct': pm_1, 'par_add_1pct': pa_1,
                'alpha_mult': am, 'alpha_add': aa
            })
    pd.DataFrame(results).to_csv("data_robustness_pareto_add.csv", index=False)
    print("\nPareto/Additive Robustness data saved.")

# =============================================================================
# 4. EXTREME DIAGNOSTICS
# =============================================================================
# =============================================================================
# 4. EXTREME DIAGNOSTICS (BIMODAL / BETA)
# =============================================================================
def compute_extreme_diagnostics(n_agents=2000000):
    print("--- COMPUTING EXTREME DIAGNOSTICS ---")
    
    # You can change Beta to 2.5 here if you want to test the "Wealthy Ghost" economy,
    # or keep it at 0.05 with Sigma 8.0 for the "Coin Flip" economy.
    BETA, SIGMA_NU, T_MEAN, T_SHARE = .05, 4.5, 65000, 0.20
    
    low, high, cal_sigma = 0.1, 8.0, 1.0
    for i in range(25):
        guess = (low + high) / 2
        np.random.seed(1234) 
        
        # 1. Explicitly create the temporary array
        yt_temp = np.random.lognormal(np.log(T_MEAN) - (guess**2 / 2), guess, 100000)
        
        # 2. Pass it to the standard function with Beta noise
        yr_temp, _ = tm.apply_evasion(yt_temp, BETA, SIGMA_NU, noise_dist='beta', seed=42)
        
        share = np.sort(yr_temp)[-1000:].sum() / yr_temp.sum()
        if abs(share - T_SHARE) < 0.002: cal_sigma = guess; break
        elif share < T_SHARE: low = guess
        else: high = guess
            
    np.random.seed(42)
    yt = np.random.lognormal(np.log(T_MEAN) - (cal_sigma**2 / 2), cal_sigma, n_agents)
    
    # 3. Generate the final 2-million agent economy with Beta noise
    yr, ev = tm.apply_evasion(yt, BETA, SIGMA_NU, noise_dist='beta', seed=43)
    
    # --- Exact Lines Math ---
    grid = np.logspace(0, -2, 50)
    idx_r, idx_t = np.argsort(yr), np.argsort(yt)
    tsort, rsort = np.sort(yt), yr[idx_r]
    ev_by_rep, ev_by_true = ev[idx_r], ev[idx_t]
    
    ts, rs, es_rep, es_true = [], [], [], []
    for p in grid:
        k = max(int(n_agents * (p/100)), 10)
        ts.append(tsort[-k:].sum() / tsort.sum())
        rs.append(rsort[-k:].sum() / rsort.sum())
        es_rep.append(ev_by_rep[-k:].mean())
        es_true.append(ev_by_true[-k:].mean())

    pd.DataFrame({'grid': grid, 'ts': ts, 'rs': rs, 'es_rep': es_rep, 'es_true': es_true}).to_csv("data_extreme_lines.csv", index=False)
    
    # --- KDE Sample ---
    pd.DataFrame({'True': yt, 'Reported': yr, 'EvasionRate': ev}).sample(20000, random_state=42).to_csv("data_extreme_kde.csv", index=False)
    print("Extreme Diagnostics data saved.")

# =============================================================================
# 5. FIXED ROBUSTNESS (TRUE & AGGREGATE)
# =============================================================================
def compute_fixed_robustness(n_agents=2000000):
    print("--- COMPUTING FIXED TRUE & FIXED AGGREGATE ROBUSTNESS ---")
    results = []
    TARGET_AGG_EV = 0.08
    baseline_sigma = tm.solve_for_reported_share('lognormal', 0.0, 0.0, 'loglinear', 'log_income', 0.20, n_agents)
    y_true_fixed = tm.generate_true_income(n_agents, 'lognormal', baseline_sigma, seed=2026)
    k1 = int(n_agents * 0.01)
    true_share_fixed = np.sort(y_true_fixed)[-k1:].sum() / y_true_fixed.sum()
    
    for beta in BETA_VALS:
        for snu in SIGMA_VALS:
            print(f"  Simulating Fixed Robustness: Beta={beta:.2f}, Sigma={snu:.1f}...", end="\r")
            
            y_rep_ft, _ = tm.apply_evasion(y_true_fixed, beta=beta, sigma_nu=snu, mode='loglinear', z_type='log_income', seed=999)
            rep_share_ft = np.sort(y_rep_ft)[-k1:].sum() / y_rep_ft.sum()
            
            best_eb = tm.solve_for_base_evasion(y_true_fixed, beta, snu, target_evasion=TARGET_AGG_EV, mode='loglinear', z_type='log_income')
            y_rep_fa, _ = tm.apply_evasion(y_true_fixed, beta, snu, mode='loglinear', base_evasion=best_eb, seed=999)
            
            results.append({
                'Beta': beta, 'Sigma': snu,
                'fixed_true_rep_share': rep_share_ft,
                'fixed_true_gap': true_share_fixed - rep_share_ft,
                'fixed_agg_ebase': best_eb,
                'fixed_agg_taxgap': (y_true_fixed.sum() - y_rep_fa.sum()) / y_true_fixed.sum(),
                'fixed_agg_repgap': true_share_fixed - (np.sort(y_rep_fa)[-k1:].sum() / y_rep_fa.sum())
            })
    pd.DataFrame(results).to_csv("data_fixed_robustness.csv", index=False)
    print("\nFixed Robustness data saved.")

# =============================================================================
# 6. BIMODAL ROBUSTNESS
# =============================================================================
def compute_bimodal_robustness(n_agents=2000000):
    print("--- COMPUTING BIMODAL ROBUSTNESS ---")
    results = []
    target_beta, target_snu = 0.05, 1.4
    
    for beta in BETA_VALS:
        for snu in SIGMA_VALS:
            print(f"  Bimodal Cell: Beta={beta:.2f}, Sigma={snu:.1f}...", end="\r")
            df, _ = tm.get_calibrated_scenario('lognormal', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=n_agents, noise_dist='beta')
            
            k = int(n_agents * 0.01)
            t1 = df.nlargest(k, 'True')['True'].sum() / df['True'].sum()
            r1 = df.nlargest(k, 'Reported')['Reported'].sum() / df['Reported'].sum()
            top1 = df.nlargest(k, 'True')
            
            alpha_val, beta_param = np.nan, np.nan
            if snu > 0:
                log_y = np.log(df['True'])
                avg_ev = np.clip(0.12 * np.exp(beta * ((log_y - log_y.mean()) / log_y.std())), 1e-5, 0.95)
                alpha_val = (avg_ev / snu).mean()
                beta_param = ((1.0 - avg_ev) / snu).mean()
            
            results.append({
                'Beta': beta, 'Sigma': snu,
                'res_map': t1 - r1,
                'rate_1pct_map': (top1['True'] - top1['Reported']).sum() / top1['True'].sum(),
                'agg_ev_map': (df['True'].sum() - df['Reported'].sum()) / df['True'].sum(),
                'alpha_map': alpha_val, 'beta_param_map': beta_param
            })
            
            if np.isclose(beta, target_beta) and np.isclose(snu, target_snu):
                # 1. Exact Lines
                grid = np.logspace(0, -2, 50)
                idx_r, idx_t = np.argsort(df['Reported'].values), np.argsort(df['True'].values)
                tsort, rsort = np.sort(df['True'].values), df['Reported'].values[idx_r]
                ev_by_rep, ev_by_true = df['EvasionRate'].values[idx_r], df['EvasionRate'].values[idx_t]
                
                lines_data = []
                for p in grid:
                    k_val = max(int(n_agents * (p/100)), 100)
                    lines_data.append({
                        'grid_pct': p,
                        'true_share': tsort[-k_val:].sum() / tsort.sum(),
                        'rep_share': rsort[-k_val:].sum() / rsort.sum(),
                        'ev_rep': ev_by_rep[-k_val:].mean(),
                        'ev_true': ev_by_true[-k_val:].mean()
                    })
                pd.DataFrame(lines_data).to_csv("data_bimodal_lines.csv", index=False)
                
                # 2. KDE Sample
                df.sample(20000, random_state=42).to_csv("data_bimodal_kde.csv", index=False)
                
    pd.DataFrame(results).to_csv("data_bimodal_grid.csv", index=False)
    print("\nBimodal data saved.")

# =============================================================================
# 7. EQUALITY LINES
# =============================================================================
def compute_equality_lines(n_agents=2000000):
    print("--- COMPUTING EQUALITY LINES ---")
    from scipy import optimize
    
    def get_reported_share(sigma_ineq, beta, sigma_nu, noise_dist):
        y_true = tm.generate_true_income(n_agents, 'lognormal', sigma_ineq, seed=42)
        y_rep, _ = tm.apply_evasion(y_true, beta, sigma_nu, mode='loglinear', noise_dist=noise_dist, seed=43)
        k = int(n_agents * 0.01)
        return (np.sort(y_rep)[-k:].sum() / y_rep.sum()) - 0.20

    def get_true_share_gap(beta, sigma_nu, noise_dist):
        calib_sigma = optimize.brentq(get_reported_share, 0.5, 3.5, args=(beta, sigma_nu, noise_dist))
        y_true = tm.generate_true_income(n_agents, 'lognormal', calib_sigma, seed=42)
        k = int(n_agents * 0.01)
        return (np.sort(y_true)[-k:].sum() / y_true.sum()) - 0.20

    sigma_nu_values = np.linspace(0, 1.6, 30)
    results = []

    for snu in sigma_nu_values:
        print(f"  Solving Equality Line for Sigma_Nu = {snu:.2f}...", end="\r")
        try: root_norm = optimize.brentq(get_true_share_gap, -0.10, 0.80, args=(snu, 'normal'))
        except ValueError: root_norm = np.nan
        try: root_beta = optimize.brentq(get_true_share_gap, -0.10, 0.80, args=(snu, 'beta'))
        except ValueError: root_beta = np.nan
        results.append({'sigma_nu': snu, 'beta_normal': root_norm, 'beta_bimodal': root_beta})

    pd.DataFrame(results).to_csv("data_equality_lines.csv", index=False)
    print("\nEquality Lines data saved.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    N_AGENTS = 2000000 
    
    #compute_core_grid(n_agents=N_AGENTS)
    #compute_walkthrough(n_agents=N_AGENTS)
    #compute_robustness_grid(n_agents=N_AGENTS)
    compute_extreme_diagnostics(n_agents=N_AGENTS)
    #compute_fixed_robustness(n_agents=N_AGENTS)
    #compute_bimodal_robustness(n_agents=N_AGENTS)
    #compute_equality_lines(n_agents=N_AGENTS)
    
    print("\n=== ALL DATA COMPUTATION COMPLETE ===")