# -*- coding: utf-8 -*-
"""
plot_figures.py
Reads pre-computed arrays and instantly plots exact, publication-ready figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def pivot_grid(df, value_col):
    """Helper to convert flat CSV into a 2D matrix for heatmaps."""
    pivot = df.pivot(index='Beta', columns='Sigma', values=value_col)
    return pivot.sort_index(ascending=False).values, pivot.index.values[::-1], pivot.columns.values

# =============================================================================
# 1. TABLE 1: DECOMPOSITION
# =============================================================================
def plot_table1():
    print("Generating Table 1: Decomposition...")
    df = pd.read_csv("data_core_grid.csv")
    scenarios = [
        ("Progressive / Low Het", 0.10, 0.4), ("Progressive / High Het", 0.10, 1.4),
        ("Regressive / Low Het", -0.05, 0.4), ("Regressive / High Het", -0.05, 1.4)
    ]
    out_list = []
    
    for name, b, s in scenarios:
        row = df[(np.isclose(df['Beta'], b)) & (np.isclose(df['Sigma'], s))].iloc[0]
        s_true, s_rep, s_rep_given_true = row['s_true'], row['s_rep'], row['s_rep_given_true']
        
        out_list.append({
            "Scenario": name, "True Share": s_true, "Reported Share": s_rep,
            "Total Gap": s_true - s_rep, "Measurement": s_true - s_rep_given_true, 
            "Re-ranking": s_rep_given_true - s_rep
        })
    
    out = pd.DataFrame(out_list)
    for col in out.columns[1:]: out[col] = out[col].apply(lambda x: f"{x*100:+.1f}%")
    print("\n" + out.to_string(index=False) + "\n")
    out.to_csv("Tab1_Decomposition.csv", index=False)

# =============================================================================
# 2. CORE HEATMAPS
# =============================================================================
def plot_all_heatmaps():
    print("Generating Heatmaps (Figs 1, 2, 3, Combined, Gini)...")
    df = pd.read_csv("data_core_grid.csv")
    
    r1, y_lbl, x_lbl = pivot_grid(df, 'rate_1pct')
    r01, _, _ = pivot_grid(df, 'rate_01pct')
    agg, _, _ = pivot_grid(df, 'agg_gap')
    g1, _, _ = pivot_grid(df, 'gap_1pct')
    g01, _, _ = pivot_grid(df, 'gap_01pct')
    gini_val, _, _ = pivot_grid(df, 'gini_diff')

    y_lbl = [f"{b:.2f}" for b in y_lbl]
    x_lbl = [f"{s:.1f}" for s in x_lbl]

    seq_style = dict(annot=True, fmt=".1%", cmap="Reds", cbar=False, xticklabels=x_lbl)
    div_style = dict(annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False, xticklabels=x_lbl)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(r1, ax=ax[0], yticklabels=y_lbl, **seq_style)
    ax[0].set_title("Avg Evasion Rate (Top 1%)"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(r01, ax=ax[1], yticklabels=False, **seq_style)
    ax[1].set_title("Avg Evasion Rate (Top 0.1%)"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_EvasionRates.pdf"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(agg, yticklabels=y_lbl, **seq_style)
    plt.title("Aggregate Tax Gap"); plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_TaxGap.pdf"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(g1, ax=ax[0], yticklabels=y_lbl, **div_style)
    ax[0].set_title("Reported Income Gap: Top 1% Share"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(g01, ax=ax[1], yticklabels=False, **div_style)
    ax[1].set_title("Reported Income Gap: Top 0.1% Share"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_ReportedGap.pdf"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(r1, ax=ax[0], yticklabels=y_lbl, **seq_style)
    ax[0].set_title("A. Avg Evasion Rate (Top 1%)"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(agg, ax=ax[1], yticklabels=False, **seq_style)
    ax[1].set_title("B. Aggregate Tax Gap"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_Combined_EvasionGap.pdf"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(gini_val, annot=True, fmt=".3f", cmap="RdBu_r", center=0, cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Gini Gap (True - Reported)"); plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_GiniGap.pdf"); plt.close()

# =============================================================================
# 3. SHARE LINES & EVASION PROFILES
# =============================================================================
def plot_share_lines():
    print("Generating Share Lines & Evasion Profiles...")
    
    def draw_panel(ax, beta, sigma, title):
        df_lines = pd.read_csv(f"data_core_lines_b{beta:.2f}_s{sigma:.1f}.csv")
        df_bins = pd.read_csv(f"data_core_bins_b{beta:.2f}_s{sigma:.1f}.csv")
        
        x_log, rep_c, true_c = df_lines['x_log'].values, df_lines['rep_c'].values, df_lines['true_c'].values
        
        ax.plot(x_log, rep_c, 'k', label='Reported (Ranked by Rep)', lw=1.5)
        ax.plot(x_log, true_c, 'purple', label='True (Ranked by True)', lw=1.5, alpha=0.8)
        ax.set_ylabel("Shares"); ax.set_ylim(0, 0.6)
        
        ax2 = ax.twinx()
        ax2.plot(df_bins['bin_centers_log'], df_bins['ev_profile'], color='#C00000', ls=':', lw=2, label='Evasion Rate')
        ax2.set_ylabel("Evasion Rate", color='#C00000'); ax2.tick_params(axis='y', labelcolor='#C00000'); ax2.set_ylim(0, 0.5)
        
        ax.set_title(title, fontsize=11, weight='bold'); ax.set_xlim(1.0, 3.0)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Top 10%", "Top 1%", "Top 0.1%"])
        ax.grid(True, linestyle='-', alpha=0.2)
        
        # EXACT ARGMIN TARGETING
        idx_1pct = np.abs(x_log - 2.0).argmin()
        idx_01pct = np.abs(x_log - 3.0).argmin()
        
        gap_1 = true_c[idx_1pct] - rep_c[idx_1pct]
        gap_01 = true_c[idx_01pct] - rep_c[idx_01pct]
        
        ax.text(0.03, 0.03, f"Gap (True - Rep):\nTop 1%: {gap_1:+.2%}\nTop 0.1%: {gap_01:+.2%}", 
                transform=ax.transAxes, fontsize=9, va='bottom', bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    draw_panel(ax[0,0], 0.10, 0.4, "Beta=0.1, Sigma=0.4")
    draw_panel(ax[0,1], 0.10, 1.4, "Beta=0.1, Sigma=1.4")
    draw_panel(ax[1,0], -0.05, 0.4, "Beta=-0.05, Sigma=0.4")
    draw_panel(ax[1,1], -0.05, 1.4, "Beta=-0.05, Sigma=1.4")
    plt.tight_layout(); plt.savefig("Fig_ShareLines.pdf"); plt.close()

    def plot_evasion(ax, beta, sigma, title):     
        df_evinc = pd.read_csv(f"data_core_evinc_b{beta:.2f}_s{sigma:.1f}.csv")
        centers, ev_true, ev_rep = df_evinc['centers'].values, df_evinc['ev_true'].values, df_evinc['ev_rep'].values
        
        ax.plot(centers, ev_true, 'b-', label='vs True Income', lw=2)
        ax.plot(centers, ev_rep, 'g--', label='vs Reported Income', lw=2)
        ax.axhline(0.05, color='grey', alpha=0.5, ls=':', label='Baseline (5%)')
        ax.set_xscale('log'); ax.set_title(title); ax.set_xlabel("Income ($)"); ax.set_ylabel("Avg Evasion Rate")
        ax.set_xlim(centers[0], centers[-1]); ax.set_ylim(0, 0.55); ax.legend()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plot_evasion(ax[0], 0.10, 1.4, "Beta=0.10, Sigma=1.4")
    plot_evasion(ax[1], -0.05, 1.4, "Beta=-0.05, Sigma=1.4")
    plt.tight_layout(); plt.savefig("Fig_EvasionProfiles.pdf"); plt.close()

# =============================================================================
# 4. WALKTHROUGH & EXTREME DIAGNOSTICS
# =============================================================================
def plot_walkthrough():
    print("Generating Walkthrough Fig...")
    df_kde = pd.read_csv("data_walkthrough_kde.csv")
    df_lines = pd.read_csv("data_walkthrough_lines.csv")
    stats = pd.read_csv("data_walkthrough_stats.csv").iloc[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    sns.kdeplot(x=np.log(df_kde['True']), ax=ax1, color='#d62728', lw=2, label='True')
    sns.kdeplot(x=np.log(df_kde['Reported']), ax=ax1, color='#1f77b4', lw=2, ls='--', label='Reported')
    
    ax1.axvline(np.log(stats['TargetMean']), color='k', alpha=0.4)
    ax1.axvline(np.log(stats['Cutoff_True']), color='#d62728', ls=':', ymax=0.6)
    ax1.axvline(np.log(stats['Cutoff_Rep']), color='#1f77b4', ls=':', ymax=0.5)
    
    txt_box = f"Top 1% Cutoff:\nTrue: ${stats['Cutoff_True']:,.0f}\nReported: ${stats['Cutoff_Rep']:,.0f}\nGap: ${stats['Cutoff_True'] - stats['Cutoff_Rep']:,.0f}"
    ax1.text(0.95, 0.70, txt_box, transform=ax1.transAxes, fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))
    ax1.legend(loc='upper left'); ax1.set_xlabel("Log Income"); ax1.set_title("A. Distributions & Cutoffs")
    
    ax2 = axes[1]; ax2t = ax2.twinx()
    grid_pct, ts, rs, es_rep, es_true = df_lines['grid_pct'].values, df_lines['ts'].values, df_lines['rs'].values, df_lines['es_rep'].values, df_lines['es_true'].values
    
    ax2.plot(grid_pct, ts, color='#d62728', lw=2.5, label='True Share')
    ax2.plot(grid_pct, rs, color='#1f77b4', lw=2.5, ls='--', label='Reported Share')
    ax2t.plot(grid_pct, es_rep, color='green', lw=2, ls=':', label='Avg Evasion (Rep Top %)')
    ax2t.plot(grid_pct, es_true, color='darkgreen', lw=1.5, ls='-.', label='Avg Evasion (True Top %)')
    
    ax2.set_xscale('log'); ax2.invert_xaxis(); ax2.set_xticks([1, 0.1, 0.01]); ax2.set_xticklabels(["1%", "0.1%", "0.01%"])
    ax2.set_xlabel("Top Percentile"); ax2.set_ylabel("Cumulative Income Share")
    ax2t.set_ylabel("Average Evasion Rate", color='green'); ax2t.tick_params(axis='y', labelcolor='green')
    
    lines, lbls = ax2.get_legend_handles_labels(); l2, lb2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines + l2, lbls + lb2, loc='lower left', fontsize=8); ax2t.set_title("B. Top Shares & Evasion Intensity")
    plt.tight_layout(); plt.savefig("Fig_Walkthrough_Clean.pdf"); plt.close()

def plot_extreme_diagnostics():
    print("Generating Extreme Diagnostics...")
    df_kde = pd.read_csv("data_extreme_kde.csv")
    df_lines = pd.read_csv("data_extreme_lines.csv")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    yt, yr = df_kde['True'].values, df_kde['Reported'].values
    ax1 = ax[0]
    sns.kdeplot(x=np.log(yt), ax=ax1, color='#d62728', lw=3, label='True')
    sns.kdeplot(x=np.log(np.maximum(yr, 1)), ax=ax1, color='#1f77b4', lw=3, ls='--', label='Reported')
    sns.kdeplot(x=np.log(np.maximum(yr, 1)), weights=np.maximum(yt-yr, 0), ax=ax1, color='purple', lw=3, ls=':', label='Unreported $')
    ax1.set_title("A. Distributions & Unreported $"); ax1.legend(loc='upper left')

    ax2, ax2t = ax[1], ax[1].twinx()
    grid, ts, rs, es_rep, es_true = df_lines['grid'].values, df_lines['ts'].values, df_lines['rs'].values, df_lines['es_rep'].values, df_lines['es_true'].values

    ax2.plot(grid, ts, color='#d62728', lw=3, label='True Share')
    ax2.plot(grid, rs, color='#1f77b4', lw=3, ls='--', label='Rep Share')
    ax2t.plot(grid, es_rep, color='green', lw=2, ls=':', label='Evasion (Top Rep %)')
    ax2t.plot(grid, es_true, color='darkgreen', lw=1.5, ls='-.', label='Evasion (Top True %)')
    
    ax2.set_xscale('log'); ax2.invert_xaxis(); ax2.set_xticks([1, 0.1, 0.01]); ax2.set_xticklabels(["1%", "0.1%", "0.01%"])
    ax2t.set_ylabel("Avg Evasion Rate", color='green'); ax2t.set_ylim(0, 1); ax2t.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax2.get_legend_handles_labels(); lines2, labels2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8); ax2.set_title("B. Top Shares & Evasion Intensity")
    plt.tight_layout(); plt.savefig("Fig_Extreme.pdf"); plt.close()

# =============================================================================
# 5. ROBUSTNESS HEATMAPS
# =============================================================================
def plot_robustness_heatmaps():
    print("Generating Robustness Heatmaps...")
    heat_kw = dict(annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False)
    
    df_par = pd.read_csv("data_robustness_pareto_add.csv")
    la_1, y_lbl, x_lbl = pivot_grid(df_par, 'log_add_1pct')
    la_01, _, _ = pivot_grid(df_par, 'log_add_01pct')
    pm_1, _, _ = pivot_grid(df_par, 'par_mult_1pct')
    pa_1, _, _ = pivot_grid(df_par, 'par_add_1pct')
    am, _, _ = pivot_grid(df_par, 'alpha_mult')
    aa, _, _ = pivot_grid(df_par, 'alpha_add')
    
    y_lbl = [f"{b:.2f}" for b in y_lbl]
    x_lbl = [f"{s:.1f}" for s in x_lbl]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(la_1, ax=ax[0], xticklabels=x_lbl, yticklabels=y_lbl, **heat_kw)
    ax[0].set_title("A. Additive: Top 1% Gap")
    sns.heatmap(la_01, ax=ax[1], xticklabels=x_lbl, yticklabels=False, **heat_kw)
    ax[1].set_title("B. Additive: Top 0.1% Gap")
    plt.tight_layout(); plt.savefig("Fig6_Robustness_Additive.pdf"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(pm_1, ax=ax[0], xticklabels=x_lbl, yticklabels=y_lbl, **heat_kw)
    ax[0].set_title("A. Pareto Multiplicative")
    sns.heatmap(pa_1, ax=ax[1], xticklabels=x_lbl, yticklabels=False, **heat_kw)
    ax[1].set_title("B. Pareto Additive")
    plt.tight_layout(); plt.savefig("Fig7_Robustness_Pareto.pdf"); plt.close()

    alpha_kw = dict(annot=True, fmt=".2f", cmap="viridis", cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(am, ax=ax[0], **alpha_kw)
    ax[0].set_title("A. Pareto Alpha (Multiplicative)"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(aa, ax=ax[1], **alpha_kw)
    ax[1].set_yticks([]); ax[1].set_title("B. Pareto Alpha (Additive)"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_ParetoAlpha_Robustness.pdf"); plt.close()

    df_fix = pd.read_csv("data_fixed_robustness.csv")
    ft_rep, _, _ = pivot_grid(df_fix, 'fixed_true_rep_share')
    ft_gap, _, _ = pivot_grid(df_fix, 'fixed_true_gap')
    fa_eb, _, _ = pivot_grid(df_fix, 'fixed_agg_ebase')
    fa_tax, _, _ = pivot_grid(df_fix, 'fixed_agg_taxgap')
    fa_rep, _, _ = pivot_grid(df_fix, 'fixed_agg_repgap')
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(ft_rep, annot=True, fmt=".1%", cmap="RdBu", center=0.20, cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Reported Top 1% Share\n(Holding True Inequality Fixed)")
    plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_FixedTrue_Robustness.pdf"); plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(ft_gap, annot=True, fmt=".1%", cmap="RdBu_r", center=0, xticklabels=x_lbl, yticklabels=y_lbl, cbar=False)
    plt.title("Gap When Holding True Population Fixed\n" r"Calibration: $\beta$ = $\sigma_{\nu}$ = 0")
    plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_Robustness_FixedTrue_Gap.pdf"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(fa_tax, annot=True, fmt=".1%", cmap="Reds", cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Aggregate Tax Gap (Anchored at 8.0%)", fontsize=13, fontweight='bold')
    plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_Robustness_FixedAgg_TaxGap.pdf"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(fa_eb, annot=True, fmt=".3f", cmap="viridis", cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Calibrated e_base parameter for 8.0% Tax Gap", fontsize=13, fontweight='bold')
    plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_Robustness_FixedAgg_eBase.pdf"); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(fa_rep, annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Reported Income Gap\n(Under Fixed 8.0% Total Evasion)", fontsize=13, fontweight='bold')
    plt.xlabel("Sigma"); plt.ylabel("Beta")
    plt.tight_layout(); plt.savefig("Fig_Robustness_FixedAgg_RepGap.pdf"); plt.close()

# =============================================================================
# 6. BIMODAL ROBUSTNESS & EQUALITY LINES
# =============================================================================
def plot_bimodal_robustness():
    print("Generating Bimodal Robustness...")
    df = pd.read_csv("data_bimodal_grid.csv")
    
    res_map, y_lbl, x_lbl = pivot_grid(df, 'res_map')
    r1_map, _, _ = pivot_grid(df, 'rate_1pct_map')
    agg_map, _, _ = pivot_grid(df, 'agg_ev_map')
    alpha_map, _, _ = pivot_grid(df, 'alpha_map')
    beta_map, _, _ = pivot_grid(df, 'beta_param_map')
    
    y_lbl = [f"{b:.2f}" for b in y_lbl]
    x_lbl = [f"{s:.1f}" for s in x_lbl]

    plt.figure(figsize=(8, 6))
    sns.heatmap(res_map, annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False, xticklabels=x_lbl, yticklabels=y_lbl)
    plt.title("Reported Income Gap\n(Bimodal Loglinear)", fontsize=13, fontweight='bold')
    plt.xlabel("Sigma (Evasion Heterogeneity)"); plt.ylabel("Beta (Progressivity)")
    plt.tight_layout(); plt.savefig("Fig_Robustness_Bimodal_RepGap.pdf"); plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    seq_style = dict(annot=True, fmt=".1%", cmap="Reds", cbar=False, xticklabels=x_lbl)
    sns.heatmap(r1_map, ax=ax[0], yticklabels=y_lbl, **seq_style)
    ax[0].set_title("A. Avg Evasion Rate (Top 1%)"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(agg_map, ax=ax[1], yticklabels=False, **seq_style)
    ax[1].set_title("B. Aggregate Tax Gap"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_Robustness_Bimodal_Evasion.pdf"); plt.close()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    param_style = dict(annot=True, fmt=".2f", cmap="viridis", cbar=False, xticklabels=x_lbl)
    sns.heatmap(alpha_map, ax=ax[0], yticklabels=y_lbl, **param_style)
    ax[0].set_title("A. Average Alpha Parameter"); ax[0].set_ylabel("Beta"); ax[0].set_xlabel("Sigma")
    sns.heatmap(beta_map, ax=ax[1], yticklabels=False, **param_style)
    ax[1].set_title("B. Average Beta Parameter"); ax[1].set_xlabel("Sigma")
    plt.tight_layout(); plt.savefig("Fig_Robustness_Bimodal_Parameters.pdf"); plt.close()
    
    try:
        df_kde = pd.read_csv("data_bimodal_kde.csv")
        df_lines = pd.read_csv("data_bimodal_lines.csv")
        yt, yr = df_kde['True'].values, df_kde['Reported'].values
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.kdeplot(x=np.log(yt), weights=yt, ax=ax[0], color='#d62728', lw=1.5, label='True $')
        sns.kdeplot(x=np.log(np.maximum(yr, 1)), weights=yr, ax=ax[0], color='#1f77b4', lw=1.5, ls='--', label='Reported $')
        sns.kdeplot(x=np.log(np.maximum(yr, 1)), weights=yt-yr, ax=ax[0], color='purple', lw=1.5, ls=':', label='Unreported $')
        sns.kdeplot(x=np.log(yt), weights=yt-yr, ax=ax[0], color='magenta', lw=1.5, ls='-.', label='Unreported $ (by True)')
        ax[0].set_title("A. Distribution of Dollars (Weighted)"); ax[0].set_ylabel("Share of Total Dollars"); ax[0].legend(loc='upper left')

        ax2 = ax[1]; ax2t = ax2.twinx()
        grid, ts, rs, es_rep, es_true = df_lines['grid_pct'].values, df_lines['true_share'].values, df_lines['rep_share'].values, df_lines['ev_rep'].values, df_lines['ev_true'].values
        
        ax2.plot(grid, ts, color='#d62728', lw=1.5, label='True Share')
        ax2.plot(grid, rs, color='#1f77b4', lw=1.5, ls='--', label='Rep Share')
        ax2t.plot(grid, es_rep, color='green', lw=1.5, ls=':', label='Evasion (Top Rep %)')
        ax2t.plot(grid, es_true, color='darkgreen', lw=1.5, ls='-.', label='Evasion (Top True %)')
        
        ax2.set_xscale('log'); ax2.invert_xaxis(); ax2.set_xticks([1, 0.1, 0.01]); ax2.set_xticklabels(["1%", "0.1%", "0.01%"])
        ax2t.set_ylabel("Avg Evasion Rate", color='green'); ax2t.set_ylim(0, 1.0); ax2t.tick_params(axis='y', labelcolor='green')
        
        lines1, labels1 = ax2.get_legend_handles_labels(); lines2, labels2 = ax2t.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        ax2.set_title("B. Top Shares & Evasion Intensity")
        fig.suptitle(f"Bimodal Beta Evasion Model ($\\beta$ = 0.05 | $\\sigma_\\nu$ = 1.4)", fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig("Fig_Robustness_Bimodal_Lines.pdf"); plt.close()
    except FileNotFoundError:
        pass

def plot_equality_lines():
    print("Generating Equality Lines...")
    try: df = pd.read_csv("data_equality_lines.csv")
    except FileNotFoundError: return

    fig, ax = plt.subplots(figsize=(8, 6))
    x_vals, y_norm_raw, y_beta_raw = df['sigma_nu'].values, df['beta_normal'].values, df['beta_bimodal'].values

    valid_norm, valid_beta = ~np.isnan(y_norm_raw), ~np.isnan(y_beta_raw)
    x_norm, y_norm = x_vals[valid_norm], y_norm_raw[valid_norm]
    x_beta, y_beta = x_vals[valid_beta], y_beta_raw[valid_beta]

    poly_func = np.poly1d(np.polyfit(x_beta, y_beta, 3))
    y_beta_smooth = poly_func(x_beta)

    ax.plot(x_norm, y_norm, label='Normal Evasion', color='#1f77b4', linewidth=3)
    ax.plot(x_beta, y_beta_smooth, label='Bimodal Beta Evasion', color='#ff7f0e', linewidth=3, linestyle='--')
    ax.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle=':')

    y_norm_extended = np.interp(x_beta, x_norm, y_norm, right=1.0)
    ax.fill_between(x_beta, -0.10, y_beta_smooth, color='blue', alpha=0.05)
    ax.fill_between(x_beta, y_beta_smooth, y_norm_extended, color='purple', alpha=0.1, label='Model Divergence Zone')
    ax.fill_between(x_beta, y_norm_extended, 1.0, color='red', alpha=0.05)

    ax.text(0.1, 0.45, "True 1% Share > Reported", fontsize=11, color='darkred', fontweight='bold')
    ax.text(1.1, 0.05, "Reported 1% Share > True", fontsize=11, color='darkblue', fontweight='bold')

    ax.set_title("Evasion Parameter Equality Lines\n(Where True Top 1% Share = Reported Top 1% Share)", fontsize=14, fontweight='bold')
    ax.set_xlabel(r"Evasion Heterogeneity ($\sigma_\nu$)"); ax.set_ylabel(r"Evasion Progressivity ($\beta$)")
    ax.set_xlim(0, 1.6); ax.set_ylim(-0.05, 0.50)
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig("Fig_Equality_Lines_Final.pdf"); plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
   # plot_table1()
   # plot_all_heatmaps()
    #plot_share_lines()
    #plot_walkthrough()
    plot_extreme_diagnostics()
    #plot_robustness_heatmaps()
    #plot_bimodal_robustness()
    #plot_equality_lines()
    print("\n=== ALL FIGURES GENERATED SUCCESSFULLY ===")