"""
CORE ENGINE: tax_model.py
Contains all logic for generating income distributions, applying evasion,
and calibrating parameters.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- CONFIGURATION ---
DEFAULT_N = 10000000
TARGET_REPORTED_SHARE = 0.20
TOLERANCE = 0.00005
MAX_EVASION = 0.9999 

def generate_true_income(n, dist_type, inequality_param, seed=None):
    if seed is not None: np.random.seed(seed)
    if dist_type == 'lognormal':
        return np.random.lognormal(mean=10.5, sigma=inequality_param, size=n)
    elif dist_type == 'pareto':
        return (np.random.pareto(inequality_param, size=n) + 1)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

def get_z_score(y_true, z_type):
    if z_type == 'log_income':
        log_y = np.log(y_true)
        if log_y.std() == 0: return np.zeros_like(log_y)
        return (log_y - log_y.mean()) / log_y.std()
    elif z_type == 'rank':
        n = len(y_true)
        ranks = np.argsort(np.argsort(y_true)) + 1
        return norm.ppf(ranks / (n + 1))
    else: raise ValueError(f"Unknown z_type: {z_type}")

def apply_evasion(y_true, beta, sigma_nu, mode='loglinear', z_type='log_income', base_evasion=0.12, noise_dist='normal', seed=None):
    if seed is not None: np.random.seed(seed)
    z = get_z_score(y_true, z_type)
    
    if noise_dist == 'normal':
        # --- BASELINE NORMAL NOISE ---
        noise = np.random.normal(0, 1, len(y_true))
        if mode == 'additive':
            raw_evasion = base_evasion + (beta * z) + (sigma_nu * noise)
        elif mode == 'loglinear':
            raw_evasion = base_evasion * np.exp((beta * z) + (sigma_nu * noise))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        evasion_rate = np.clip(raw_evasion, 0.0, MAX_EVASION)
        
    elif noise_dist == 'beta':      
        if mode == 'additive':
            avg_evasion = base_evasion + (beta * z)
        elif mode == 'loglinear':
            avg_evasion = base_evasion * np.exp(beta * z)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Clip to ensure valid Beta parameters (>0) and respect max evasion
        avg_evasion = np.clip(avg_evasion, 1e-5, MAX_EVASION)
        
        if sigma_nu > 0:
            # sigma_nu sets the dispersion. Larger sigma_nu = more U-shaped.
            xalpha = avg_evasion / sigma_nu
            xbeta = (1.0 - avg_evasion) / sigma_nu
            evasion_rate = np.random.beta(xalpha, xbeta, len(y_true))
        else:
            evasion_rate = avg_evasion
            
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")
    
    y_reported = y_true * (1 - evasion_rate)
    return y_reported, evasion_rate

def apply_evasion_extreme(y_true, beta, sigma_nu, seed=None):
    if seed is not None: np.random.seed(seed)
    
    log_y = np.log(y_true)
    if log_y.std() == 0: 
        z = np.zeros_like(log_y)
    else: 
        z = (log_y - log_y.mean()) / log_y.std()
        
    noise = np.random.normal(0, 1, len(y_true))
    base_evasion = 0.05 
    raw_evasion = base_evasion * np.exp((beta * z) + (sigma_nu * noise))
    
    # 2. UPDATE EXTREME FUNCTION to use MAX_EVASION
    evasion_rate = np.clip(raw_evasion, 0.0, MAX_EVASION)
    y_rep = y_true * (1 - evasion_rate)
    
    return y_rep, evasion_rate

def solve_for_reported_share(dist_type, beta, sigma_nu, mode, z_type, target=TARGET_REPORTED_SHARE, n_agents=DEFAULT_N, seed=42, noise_dist='normal'):
    if dist_type == 'lognormal': 
        low, high = 0.1, 4.0
    else: 
        low, high = 1.001, 8.0 
        
    guess = (low + high) / 2
    
    for i in range(50):
        guess = (low + high) / 2
        y_true = generate_true_income(n_agents, dist_type, guess, seed=seed)
        y_rep, _ = apply_evasion(y_true, beta, sigma_nu, mode=mode, z_type=z_type, noise_dist=noise_dist, seed=seed+1)
        k = int(n_agents * 0.01)
        rep_share = np.sort(y_rep)[-k:].sum() / y_rep.sum()
        
        if abs(rep_share - target) < TOLERANCE:
            return guess
        
        if dist_type == 'lognormal':
            if rep_share < target: low = guess
            else: high = guess
        else:
            if rep_share < target: high = guess 
            else: low = guess
            
    return guess

def solve_for_base_evasion(y_true, beta, sigma_nu, target_evasion=0.08, mode='loglinear', z_type='log_income', noise_dist='normal'):
    low, high = 0.0001, 0.50
    guess = 0.05
    
    for _ in range(30):
        guess = (low + high) / 2
        y_rep, _ = apply_evasion(y_true, beta, sigma_nu, mode=mode, z_type=z_type, base_evasion=guess, noise_dist=noise_dist, seed=999)
        agg_ev = (y_true.sum() - y_rep.sum()) / y_true.sum()
        
        if abs(agg_ev - target_evasion) < 0.0001:
            return guess
        elif agg_ev < target_evasion:
            low = guess
        else:
            high = guess
            
    return guess

def get_calibrated_scenario(dist_type, beta, sigma_nu, mode='loglinear', z_type='log_income', n_agents=DEFAULT_N, seed=1000, noise_dist='normal'):
    calibrated_param = solve_for_reported_share(dist_type, beta, sigma_nu, mode, z_type, n_agents=n_agents, seed=seed, noise_dist=noise_dist)
    y_true = generate_true_income(n_agents, dist_type, calibrated_param, seed=seed)
    y_rep, ev_rate = apply_evasion(y_true, beta, sigma_nu, mode=mode, z_type=z_type, noise_dist=noise_dist, seed=seed+1)
    return pd.DataFrame({'True': y_true, 'Reported': y_rep, 'EvasionRate': ev_rate}), calibrated_param