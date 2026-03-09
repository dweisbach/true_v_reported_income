# Tax Evasion Simulation & Analysis

This project simulates income tax evasion behavior under varying levels of tax progressivity ($\beta$) and evasion heterogeneity ($\sigma$). It provides a framework for analyzing the gap between "True" and "Reported" income inequality statistics.

## 🚀 Overview
The simulation engine models how reporting behavior shifts measured inequality, allowing researchers to decompose the "Inequality Gap" into measurement error and agent re-ranking effects.

### 📂 Project Structure
* **Tax_Model.py**: This is the backend computationo engine
* **Compute_data.py**: Produces CSVs with the data for plotting
* **plot_figures.py**: Plots stuff


## 📊 List of Outputs
TBD

| File | Description |
| :--- | :--- |
| **Fig_EvasionRates.pdf** | Heatmap of evasion rates for the Top 1% and 0.1%. |
| **Fig_TaxGap.pdf** | Heatmap of the aggregate tax gap. |
| **Fig_ReportedGap.pdf** | Heatmap of the "Inequality Gap" (True - Reported). |
| **Fig_ShareLines.pdf** | Comparison of Reported vs. True shares (Top 10% to 0.1%). |
| **Fig_EvasionProfiles.pdf** | Curves showing evasion rates vs. income levels. |
| **Fig_Robustness_Additive.pdf**| Robustness check using Additive evasion logic. |
| **Fig_Robustness_Pareto.pdf** | Robustness check using Pareto distributions. |
| **Fig_Walkthrough_Clean.pdf** | Summary density plot and selection effect analysis. |
| **Fig_GiniGap.pdf** | Heatmap of the difference between True and Reported Gini. |
| **Tab1_Decomposition.csv** | Table showing Measurement vs. Re-ranking effects. |
| **Fig_Robustness_FixedTrue_Gap.pdf** |Heatmap of reported income gap holding true fixed |
| **Fig_FixedTrue_Robustness.pdf** | Heatmap of reported 1% shares, holding true fixed |

## ⚙️ Setup & Requirements
You need Python 3.x installed along with the following libraries:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

Install them via command line:
```bash

pip install numpy pandas matplotlib seaborn scipy
