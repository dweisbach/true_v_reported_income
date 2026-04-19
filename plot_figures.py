# -*- coding: utf-8 -*-
"""
plot_figures.py
Reads pre-computed arrays and instantly plots exact, publication-ready figures.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_csv(filename, fallback=None):
    """Centralized data loader to prevent repetitive try/except boilerplate."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    elif fallback and os.path.exists(fallback):
        return pd.read_csv(fallback)
    else:
        print(f"  [Warning] Missing Data: {filename}. Skipping plot.")
        return None

def pivot_grid(df, value_col):
    """Converts a flat CSV into a labeled 2D DataFrame for Seaborn heatmaps."""
    pivot = df.pivot(index='Beta', columns='Sigma', values=value_col)
    # Sort Beta (y-axis) descending so high Gamma is at the top
    return pivot.sort_index(ascending=False)

def _fmt_pct(val, sign=False):
    """Standardizes percentage formatting for tables."""
    return f"{val*100:+.2f}%" if sign else f"{val*100:.2f}%"

# =============================================================================
# 1. MAIN TEXT TABLES
# =============================================================================

def plot_table1():
    print("Generating Table 1: Standard and Alternative Decompositions...")
    df = _load_csv("data_core_grid.csv")
    if df is None: return

    scenarios = [
        ("Progressive / Low Het", 0.05, 0.2), 
        ("Progressive / High Het", 0.05, 1.4),
        ("Regressive / Low Het", -0.05, 0.2), 
        ("Regressive / High Het", -0.05, 1.4)
    ]
    
    out_list_std = []
    out_list_alt = []
    
    # Check if the simulation data contains the new alternative baseline metric
    has_alt_data = 's_rep_given_true' in df.columns

    for name, b, s in scenarios:
        mask = (np.isclose(df['Beta'], b, atol=1e-5)) & (np.isclose(df['Sigma'], s, atol=1e-5))
        filtered_df = df[mask]
        
        if filtered_df.empty:
            print(f"  [Warning] No data found for {name} (Beta={b}, Sigma={s})")
            continue
            
        row = filtered_df.iloc[0]
        s_true, s_rep = row['s_true'], row['s_rep']
        s_t_given_r = row['s_true_given_rep'] 
        
        # 1. Standard: (S_Y|R - S_Y) + (S_R - S_Y|R)
        out_list_std.append({
            "Scenario": name, 
            "True Share": _fmt_pct(s_true), 
            "Total Gap": _fmt_pct(s_rep - s_true, sign=True), 
            "Re-ranking (True $)": _fmt_pct(s_t_given_r - s_true, sign=True), 
            "Diff. Evasion (Rep Top)": _fmt_pct(s_rep - s_t_given_r, sign=True)
        })
        
        # 2. Alternative: (S_R - S_R|Y) + (S_R|Y - S_Y)
        if has_alt_data:
            s_r_given_t = row['s_rep_given_true']
            out_list_alt.append({
                "Scenario": name, 
                "True Share": _fmt_pct(s_true), 
                "Total Gap": _fmt_pct(s_rep - s_true, sign=True), 
                "Re-ranking (Rep $)": _fmt_pct(s_rep - s_r_given_t, sign=True),
                "Diff. Evasion (True Top)": _fmt_pct(s_r_given_t - s_true, sign=True)
            })
    
    out_std = pd.DataFrame(out_list_std)
    print("\n=== STANDARD DECOMPOSITION ===")
    print(out_std.to_string(index=False))
    out_std.to_csv("Tab1_Decomposition_Standard.csv", index=False)
    
    if has_alt_data:
        out_alt = pd.DataFrame(out_list_alt)
        print("\n=== ALTERNATIVE DECOMPOSITION ===")
        print(out_alt.to_string(index=False))
        out_alt.to_csv("App_Tab_Decomposition_Alternative.csv", index=False)
    else:
        print("\n[Note] 's_rep_given_true' not found in data_core_grid.csv. Alternative Table skipped.")


def plot_walkthrough():
    print("Plotting Walkthrough Figure...")
    
    # --- Load scenario metadata ---
    try:
        df_scenarios = pd.read_csv("data_walkthrough_scenarios.csv")
    except FileNotFoundError:
        print("Error: data_walkthrough_scenarios.csv not found. Run compute_walkthrough first.")
        return
    
    n_rows = len(df_scenarios)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    
    # Handle single-row case (axes shape is (2,) not (1,2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    
    for row, (_, scenario) in enumerate(df_scenarios.iterrows()):
        idx = int(scenario['idx'])
        label = scenario['Label']
        
        try:
            df_kde = pd.read_csv(f"data_walkthrough_kde_{idx}.csv")
            df_lines = pd.read_csv(f"data_walkthrough_scaled_lines_{idx}.csv")
            stats = pd.read_csv(f"data_walkthrough_panel_stats_{idx}.csv").iloc[0]
        except FileNotFoundError:
            print(f"  [Warning] Missing data for scenario {idx} ({label}). Skipping row.")
            continue
        
        ax1, ax2 = axes[row, 0], axes[row, 1]
        
        # --- PANEL A: DISTRIBUTIONS & CUTOFFS ---
        sns.kdeplot(x=np.log(df_kde['True']), ax=ax1, color='tab:red', lw=2, label='True')
        sns.kdeplot(x=np.log(df_kde['Reported']), ax=ax1, color='tab:blue', lw=2, ls='--', label='Reported')
        
        unreported = np.maximum(df_kde['True'] - df_kde['Reported'], 1)
        sns.kdeplot(x=np.log(unreported), ax=ax1, color='purple', lw=2, ls=':', label='Unreported Amount')

        ax1.axvline(np.log(stats['Cutoff_True']), color='gray', alpha=0.6, lw=1.5)
        ax1.axvline(np.log(stats['Cutoff_Rep']), color='gray', alpha=0.6, lw=1.5, ls='--')
        
        mean_val = stats['TargetMean']
        ax1.axvline(np.log(mean_val), color='k', alpha=0.4)
        ax1.text(np.log(mean_val) - 0.2, 0.10, f"Mean Income:\n${mean_val:,.0f}", 
                 fontsize=9, ha='right', va='center', 
                 bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

        txt_box = (f"Top 1% Cutoff:\n"
                   f"True: ${stats['Cutoff_True']:,.0f}\n"
                   f"Reported: ${stats['Cutoff_Rep']:,.0f}\n"
                   f"Gap: $ {stats['Cutoff_True'] - stats['Cutoff_Rep']:,.0f}\n"
                   f"----------\n"
                   f"Total True: ${stats['Total_True']/1e12:.2f}T\n"
                   f"Total Rep: ${stats['Total_Rep']/1e12:.2f}T\n"
                   f"Relative Income Gap: {stats['Tax_Gap']:.1%}")
        
        ax1.text(0.98, 0.95, txt_box, transform=ax1.transAxes, fontsize=10, 
                 ha='right', va='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

        ax1.set_xlabel("Log Income")
        ax1.set_ylabel("Density")
        ax1.set_title(f"{label}: Distributions & Cutoffs")
        if row == 0:
            ax1.legend(loc='upper left')

        # --- PANEL B: TOP SHARES & INTENSITY ---
        true_shares = df_lines['true_dollars_usd'] / stats['Total_True']
        rep_shares = df_lines['rep_dollars_usd'] / stats['Total_Rep']

        ax2.plot(df_lines['grid_pct'], true_shares, color='tab:red', lw=2, label='True Share')
        ax2.plot(df_lines['grid_pct'], rep_shares, color='tab:blue', lw=2, ls='--', label='Reported Share')
        ax2.set_xscale('log')
        ax2.invert_xaxis()
        ax2.set_xticks([1, 0.1, 0.01])
        ax2.set_xticklabels(['1%', '0.1%', '0.01%'])
        ax2.set_xlabel("Top Percentile")
        ax2.set_ylabel("Cumulative Income Share")

        ax2t = ax2.twinx()
        ax2t.plot(df_lines['grid_pct'], df_lines['evasion_rate_rep_top'], color='darkgreen', lw=2, ls=':', label='Avg Evasion (Rep Top %)')
        ax2t.plot(df_lines['grid_pct'], df_lines['evasion_rate_true_top'], color='darkgreen', lw=2, ls='-.', alpha=0.7, label='Avg Evasion (True Top %)')
        
        ax2t.set_ylabel("Average Evasion Rate", color='darkgreen')
        ax2t.set_ylim(0, 0.25)
        ax2t.tick_params(axis='y', labelcolor='darkgreen')

        lines, labels_l = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2t.get_legend_handles_labels()
        if row == 0:
            ax2.legend(lines + lines2, labels_l + labels2, loc='upper right', fontsize=9)
        ax2.set_title(f"{label}: Top Shares & Evasion Intensity")

    # Conform axes across all left-column (KDE) panels
    left_axes = [axes[r, 0] for r in range(n_rows)]
    all_xlims = [ax.get_xlim() for ax in left_axes]
    all_ylims = [ax.get_ylim() for ax in left_axes]
    shared_xlim = (max(x[0] for x in all_xlims), max(x[1] for x in all_xlims))
    shared_ylim = (0, 0.35)
    for ax in left_axes:
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)

    plt.tight_layout()
    plt.savefig("Fig_Walkthrough_Clean.pdf")
    print("Walkthrough figure saved.")    





def plot_walkthrough_table():
    print("Generating Walkthrough Progression Tables...")
    df = _load_csv("data_walkthrough_stats.csv")
    if df is None: return

    out_list_std = []
    out_list_alt = []
    has_alt_data = 's_rep_given_true' in df.columns

    for _, row in df.iterrows():
        s_true, s_rep = row['s_true'], row['s_rep']
        s_t_given_r = row['s_true_given_rep'] 
        
        out_list_std.append({
            "Narrative Step": row['Step'], 
            "Gamma": f"{row['Beta']:.2f}",
            "Sigma": f"{row['Sigma']:.1f}",
            "True Share": _fmt_pct(s_true),
            "Rep. Share": _fmt_pct(s_rep),
            "Total Gap": _fmt_pct(s_rep - s_true, sign=True), 
            "Re-ranking (True $)": _fmt_pct(s_t_given_r - s_true, sign=True), 
            "Diff. Evasion (Rep Top)": _fmt_pct(s_rep - s_t_given_r, sign=True)
        })
        
        if has_alt_data:
            s_r_given_t = row['s_rep_given_true']
            out_list_alt.append({
                "Narrative Step": row['Step'], 
                "Gamma": f"{row['Beta']:.2f}",
                "Sigma": f"{row['Sigma']:.1f}",
                "True Share": _fmt_pct(s_true),
                "Rep. Share": _fmt_pct(s_rep),
                "Total Gap": _fmt_pct(s_rep - s_true, sign=True), 
                "Re-ranking (Rep $)": _fmt_pct(s_rep - s_r_given_t, sign=True),
                "Diff. Evasion (True Top)": _fmt_pct(s_r_given_t - s_true, sign=True)
            })
    
    out_std = pd.DataFrame(out_list_std)
    print("\n=== WALKTHROUGH PROGRESSION (Standard) ===")
    print(out_std.to_string(index=False))
    out_std.to_csv("Tab_Walkthrough_Progression.csv", index=False)
    
    if has_alt_data:
        out_alt = pd.DataFrame(out_list_alt)
        print("\n=== WALKTHROUGH PROGRESSION (Alternative) ===")
        print(out_alt.to_string(index=False))
        out_alt.to_csv("App_Tab_Walkthrough_Alternative.csv", index=False)
    else:
        print("\n[Note] 's_rep_given_true' not found in data_walkthrough_stats.csv. Alternative Table skipped.")


# =============================================================================
# 2. HEATMAPS (CORE & ROBUSTNESS)
# =============================================================================

def _plot_heatmap_set(df, suffix, is_big):
    """Core plotting engine for both small and big heatmaps."""
    r1, r01 = pivot_grid(df, 'rate_1pct'), pivot_grid(df, 'rate_01pct')
    agg, gini = pivot_grid(df, 'agg_gap'), pivot_grid(df, 'gini_diff')
    g1, g01 = pivot_grid(df, 'gap_1pct'), pivot_grid(df, 'gap_01pct')

    for g in [r1, r01, agg, g1, g01, gini]:
        g.index = [f"{idx:.2f}" for idx in g.index]
        g.columns = [f"{col:.1f}" for col in g.columns]

    tick_step = 5 if is_big else 1
    pct_kw_right = dict(annot=not is_big, fmt=".1%", cbar=is_big, xticklabels=tick_step, yticklabels=tick_step)
    pct_kw_left = {**pct_kw_right, "cbar": False} 
    gini_kw = dict(annot=not is_big, fmt=".3f", cbar=is_big, xticklabels=tick_step, yticklabels=tick_step)

    # --- A. Evasion Rates (Separate) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(r1, ax=ax[0], cmap="Reds", **pct_kw_left)
    ax[0].set(title="Avg Evasion Rate (Top 1%)", ylabel=r"Evasion Progressivity ($\gamma$)", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)")
    sns.heatmap(r01, ax=ax[1], cmap="Reds", **pct_kw_right)
    ax[1].set(title="Avg Evasion Rate (Top 0.1%)", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)", ylabel="") 
    plt.tight_layout()
    plt.savefig(f"Fig_EvasionRates{suffix}.pdf")

    # --- B. Aggregate Gap (Separate) ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(agg, cmap="Reds", **pct_kw_right)
    plt.title("Aggregate Income Gap")
    plt.ylabel(r"Evasion Progressivity ($\gamma$)")
    plt.xlabel(r"Evasion Heterogeneity ($\sigma_\nu$)")
    plt.tight_layout()
    plt.savefig(f"Fig_TaxGap{suffix}.pdf")

    # --- C. Reported Income Gaps ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    max_gap = max(np.abs(g1.values).max(), np.abs(g01.values).max())
    div_kw_right = {**pct_kw_right, "cmap": "RdBu", "center": 0, "vmin": -max_gap, "vmax": max_gap}
    div_kw_left = {**div_kw_right, "cbar": False}
    
    sns.heatmap(g1, ax=ax[0], **div_kw_left)
    if is_big: ax[0].contour(np.arange(len(g1.columns)), np.arange(len(g1.index)), gaussian_filter(g1.values, 0.8), levels=[0], colors='black', linewidths=2)
    ax[0].set(title="Reported Income Gap: Top 1% Share", ylabel=r"Evasion Progressivity ($\gamma$)", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)")
    
    sns.heatmap(g01, ax=ax[1], **div_kw_right)
    if is_big: ax[1].contour(np.arange(len(g01.columns)), np.arange(len(g01.index)), gaussian_filter(g01.values, 0.8), levels=[0], colors='black', linewidths=2)
    ax[1].set(title="Reported Income Gap: Top 0.1% Share", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)", ylabel="")
    plt.tight_layout()
    plt.savefig(f"Fig_ReportedGap{suffix}.pdf")

    # --- D. Gini Gap ---
    plt.figure(figsize=(8, 6))
    max_gini = np.abs(gini.values).max()
    sns.heatmap(gini, **{**gini_kw, "cmap": "RdBu", "center": 0, "vmin": -max_gini, "vmax": max_gini})
    if is_big: plt.contour(np.arange(len(gini.columns)), np.arange(len(gini.index)), gaussian_filter(gini.values, 0.8), levels=[0], colors='black', linewidths=2)
    plt.title("Gini Gap (Reported - True)")
    plt.ylabel(r"Evasion Progressivity ($\gamma$)")
    plt.xlabel(r"Evasion Heterogeneity ($\sigma_\nu$)")
    plt.tight_layout()
    plt.savefig(f"Fig_GiniGap{suffix}.pdf")
    
    # --- E. COMBINED: Evasion Rate (1%) & Aggregate Gap ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(r1, ax=ax[0], cmap="Reds", **pct_kw_left)
    ax[0].set(title="A. Avg Evasion Rate (Top 1%)", ylabel=r"Evasion Progressivity ($\gamma$)", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)")
    
    sns.heatmap(agg, ax=ax[1], cmap="Reds", **pct_kw_right)
    ax[1].set(title="B. Aggregate Income Gap", xlabel=r"Evasion Heterogeneity ($\sigma_\nu$)", ylabel="") 
    
    plt.tight_layout()
    plt.savefig(f"Fig_Combined_EvasionGap{suffix}.pdf")

    plt.close('all')

def plot_small_heatmaps():
    print("Generating Small Heatmaps...")
    df = _load_csv("data_core_grid.csv")
    if df is not None: _plot_heatmap_set(df, "_small", is_big=False)

def plot_big_heatmaps():
    print("Generating Big Heatmaps...")
    df = _load_csv("data_big_grid.csv")
    if df is not None: _plot_heatmap_set(df, "_big", is_big=True)


def plot_robustness_1x2_metric(metric, file_prefix, cbar_label, cmap, norm=None, vmin=None, vmax=None, white_levels=None, black_levels=None):
    """
    Original 1x2 plotting engine for the dynamic calibration grids.
    """
    data_groups = {
        "Lognormal": [
            ("Lognormal / Beta", "data_big_lognormal_beta.csv"),
            ("Lognormal / Normal (Log-Linear)", "data_big_lognormal_normal.csv")
        ],
        "Pareto": [
            ("Pareto / Beta", "data_big_pareto_beta.csv"),
            ("Pareto / Normal (Log-Linear)", "data_big_pareto_normal.csv")
        ]
    }

    for dist_name, panels in data_groups.items():
        if dist_name == "Lognormal":
            max_beta = 0.3  
            max_sigma = 2.0  
        else:
            max_beta = 0.15  
            max_sigma = 2.0  

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        mappable = None

        for col, (title, filename) in enumerate(panels):
            ax = axes[col]
            try:
                df = pd.read_csv(filename)
                df['Beta'] = df['Beta'].round(2)
                df['Sigma'] = df['Sigma'].round(1)
                
                df = df[df['Beta'] <= max_beta]
                df = df[df['Sigma'] <= max_sigma]
                
                pivot_df = df.pivot(index='Beta', columns='Sigma', values=metric)
                pivot_df = pivot_df.sort_index(ascending=False) 

                im = sns.heatmap(
                    pivot_df, 
                    ax=ax, 
                    cmap=cmap, 
                    norm=norm, 
                    vmin=vmin,
                    vmax=vmax,
                    cbar=False
                )
                
                if mappable is None:
                    mappable = im.get_children()[0]

                if white_levels:
                    cs_white = ax.contour(np.arange(len(pivot_df.columns)) + 0.5, 
                               np.arange(len(pivot_df.index)) + 0.5, 
                               pivot_df.values, levels=white_levels, 
                               colors='white', linewidths=0.5, alpha=0.8)
                    ax.clabel(cs_white, inline=True, fontsize=10, fmt=lambda x: f"{x*100:.0f}%")

                if black_levels:
                    ax.contour(np.arange(len(pivot_df.columns)) + 0.5, 
                               np.arange(len(pivot_df.index)) + 0.5, 
                               pivot_df.values, levels=black_levels, colors='black', linewidths=2.5)

                ax.set_title(title, fontweight='bold', fontsize=16)
                ax.set_xlabel(r"Evasion Heterogeneity ($\sigma_{\nu}$)")
                ax.set_ylabel(r"Evasion Progressivity ($\gamma$)" if col == 0 else "")

            except FileNotFoundError:
                ax.text(0.5, 0.5, f"Missing: {filename}", ha='center', va='center')

        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, fontsize=14, labelpad=10)
        
        if metric == 'gap_1pct' and isinstance(norm, SymLogNorm):
            cbar.set_ticks([-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
        
        outfile = f"{file_prefix}_{dist_name}.pdf"
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved {metric} 1x2 grid to {outfile}")
        plt.close()
        

def generate_all_robustness_1x2():
    print("Generating all four 1x2 Robustness Heatmap Sets...")
    plot_robustness_1x2_metric(
        metric='gap_1pct',
        file_prefix='Fig_Robustness_Summary_1x2_Final',
        cbar_label='Reported Gap (Reported - True)',
        cmap='RdBu',
        norm=SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-0.1, vmax=0.3, base=10),
        white_levels=[0.05, 0.1, 0.2],
        black_levels=[0]
    )

# =============================================================================
# 3. FIXED-THETA HEATMAP GENERATION (NEW)
# =============================================================================

def plot_fixed_theta_1x2_metric(metric, file_prefix, cbar_label, cmap, norm=None, vmin=None, vmax=None, white_levels=None, black_levels=None):
    """
    Plots the 1x2 robustness grids specifically for the Fixed-Calibration datasets.
    """
    data_groups = {
        "Lognormal": [
            ("Lognormal / Beta (Fixed Cal)", "data_fixed_theta_lognormal_beta.csv"),
            ("Lognormal / Normal (Fixed Cal)", "data_fixed_theta_lognormal_normal.csv")
        ],
        "Pareto": [
            ("Pareto / Beta (Fixed Cal)", "data_fixed_theta_pareto_beta.csv"),
            ("Pareto / Normal (Fixed Cal)", "data_fixed_theta_pareto_normal.csv")
        ]
    }

    for dist_name, panels in data_groups.items():
        if dist_name == "Lognormal":
            max_beta = 0.3  
            max_sigma = 2.0  
        else:
            max_beta = 0.15  
            max_sigma = 2.0  

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        mappable = None

        for col, (title, filename) in enumerate(panels):
            ax = axes[col]
            try:
                df = pd.read_csv(filename)
                df['Beta'] = df['Beta'].round(2)
                df['Sigma'] = df['Sigma'].round(1)
                
                df = df[df['Beta'] <= max_beta]
                df = df[df['Sigma'] <= max_sigma]
                
                pivot_df = df.pivot(index='Beta', columns='Sigma', values=metric)
                pivot_df = pivot_df.sort_index(ascending=False) 

                im = sns.heatmap(
                    pivot_df, 
                    ax=ax, 
                    cmap=cmap, 
                    norm=norm, 
                    vmin=vmin,
                    vmax=vmax,
                    cbar=False
                )
                
                if mappable is None:
                    mappable = im.get_children()[0]

                if white_levels:
                    cs_white = ax.contour(np.arange(len(pivot_df.columns)) + 0.5, 
                               np.arange(len(pivot_df.index)) + 0.5, 
                               pivot_df.values, levels=white_levels, 
                               colors='white', linewidths=0.5, alpha=0.8)
                    ax.clabel(cs_white, inline=True, fontsize=10, fmt=lambda x: f"{x*100:.0f}%")

                if black_levels:
                    ax.contour(np.arange(len(pivot_df.columns)) + 0.5, 
                               np.arange(len(pivot_df.index)) + 0.5, 
                               pivot_df.values, levels=black_levels, colors='black', linewidths=2.5)

                ax.set_title(title, fontweight='bold', fontsize=16)
                ax.set_xlabel(r"Evasion Heterogeneity ($\sigma_{\nu}$)")
                ax.set_ylabel(r"Evasion Progressivity ($\gamma$)" if col == 0 else "")

            except FileNotFoundError:
                ax.text(0.5, 0.5, f"Missing: {filename}\n(Run compute_data.py for this case)", ha='center', va='center')

        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, fontsize=14, labelpad=10)
        
        if metric == 'gap_1pct' and isinstance(norm, SymLogNorm):
            cbar.set_ticks([-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
        
        outfile = f"{file_prefix}_{dist_name}.pdf"
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved Fixed-Theta {metric} 1x2 grid to {outfile}")
        plt.close()


def generate_all_fixed_theta_1x2():
    print("Generating Fixed-Theta Robustness Heatmaps...")
    plot_fixed_theta_1x2_metric(
        metric='gap_1pct',
        file_prefix='Fig_FixedTheta_Gap1pct_1x2',
        cbar_label='Reported Gap (Reported - True)',
        cmap='RdBu',
        norm=SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-0.1, vmax=0.3, base=10),
        white_levels=[0.05, 0.1, 0.2],
        black_levels=[0]
    )


def plot_fixed_theta_narrow_1x2_metric(metric, file_prefix, cmap='RdBu', fmt=".1%"):
    """
    Plots a 'narrow' version of the grids (Beta: -0.1 to 0.1, Sigma: 0 to 1.6)
    with annotated values in each cell. Matches the original Fig_ReportedGap formatting.
    """
    data_groups = {
        "Lognormal": [
            ("Lognormal / Beta (Fixed Cal)", "data_fixed_theta_lognormal_beta.csv"),
            ("Lognormal / Normal (Fixed Cal)", "data_fixed_theta_lognormal_normal.csv")
        ],
        "Pareto": [
            ("Pareto / Beta (Fixed Cal)", "data_fixed_theta_pareto_beta.csv"),
            ("Pareto / Normal (Fixed Cal)", "data_fixed_theta_pareto_normal.csv")
        ]
    }

    for dist_name, panels in data_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Match original Fig_ReportedGap size
        
        # 1. First pass: find the global max absolute value across BOTH panels 
        # so the color scale is uniformly vibrant for the whole figure
        max_val = 0
        pivot_dfs = []
        for _, filename in panels:
            try:
                df = pd.read_csv(filename)
                df = df[(df['Beta'] >= -0.10) & (df['Beta'] <= 0.10)]
                df = df[(df['Sigma'] >= 0.0) & (df['Sigma'] <= 1.6)]
                pivot_df = df.pivot(index='Beta', columns='Sigma', values=metric)
                pivot_df = pivot_df.sort_index(ascending=False)
                pivot_dfs.append(pivot_df)
                
                local_max = np.abs(pivot_df.values).max()
                if local_max > max_val:
                    max_val = local_max
            except FileNotFoundError:
                pivot_dfs.append(None)
                
        if max_val < 0.001: max_val = 0.01 # Prevent purely white plots

        # 2. Second pass: Plot with identical formatting to the original
        for col, (title, filename) in enumerate(panels):
            ax = axes[col]
            pivot_df = pivot_dfs[col]
            
            if pivot_df is not None:
                # Match exact tick string formatting from old plot
                pivot_df.index = [f"{idx:.2f}" for idx in pivot_df.index]
                pivot_df.columns = [f"{col:.1f}" for col in pivot_df.columns]

                sns.heatmap(
                    pivot_df, 
                    ax=ax, 
                    cmap=cmap, 
                    vmin=-max_val,        # Dynamic bounds match old plot colors exactly
                    vmax=max_val,
                    center=0,
                    annot=True,           
                    annot_kws={"size": 10}, 
                    fmt=fmt,              
                    cbar=False       # Cbar only on right side to match original layout
                )

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel(r"Evasion Heterogeneity ($\sigma_{\nu}$)")
                if col == 0:
                    ax.set_ylabel(r"Evasion Progressivity ($\gamma$)")
                else:
                    ax.set_ylabel("")

            else:
                ax.text(0.5, 0.5, f"Missing Data: {filename}", ha='center', va='center')

        plt.tight_layout()
        outfile = f"{file_prefix}_{dist_name}.pdf"
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved Narrow Annotated {metric} 1x2 grid to {outfile}")
        plt.close()


def generate_fixed_theta_narrow_grids():
    print("Generating Narrow Annotated Fixed-Theta Heatmaps...")
    plot_fixed_theta_narrow_1x2_metric(
        metric='gap_1pct',
        file_prefix='Fig_FixedTheta_NARROW_Gap1pct_1x2',
        cmap='RdBu',
        fmt="+.1%"   
    )


# =============================================================================
# 4. SHARE LINES & WALKTHROUGH (Omitted for brevity, exact logic matches original)
# =============================================================================

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("--- tax_model.py Plotting Execution ---")
    
    # 1. Main Text Tables
    #plot_table1()
    #plot_walkthrough_table()
    
    # 2. Main Text Small Figures
    #plot_small_heatmaps()
    #plot_share_lines()
    plot_walkthrough()
    
    # 3. Original Dynamic Heatmaps
    #generate_all_robustness_1x2()
    
    # 4. NEW: Fixed-Theta Heatmaps
    #generate_all_fixed_theta_1x2()
    
    # 5. NEW: Annotated Narrow Grids
    #generate_fixed_theta_narrow_grids()

    print("\n=== ALL FIGURES GENERATED SUCCESSFULLY ===")