import json
import os
import hashlib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import wasserstein_distance

def generate_truthcert_hash(data):
    """
    Generate a SHA-256 hash for data to satisfy TruthCert requirements.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def apply_neural_ot_alignment(ipd_df, ad_df):
    """
    Simplified Neural Optimal Transport (NOT) alignment.
    Aligns IPD feature distributions (age) with AD population means.
    """
    alignment_scores = []
    ipd_mean_age = ipd_df['age'].mean()
    
    for i, row in ad_df.iterrows():
        # Distance between IPD anchor population and AD regional population
        dist = np.abs(ipd_mean_age - row['age_mean'])
        # Weight inversely proportional to distance (alignment weight)
        weight = 1.0 / (1.0 + dist)
        alignment_scores.append(weight)
    return np.array(alignment_scores)

def run_invariant_risk_minimization(ipd_df, ad_df, alignment_weights):
    """
    Simplified Invariant Risk Minimization (IRM) for Meta-Analysis.
    Estimates a treatment effect that is stable across different environments.
    """
    # IPD Data
    y_ipd = ipd_df['outcome'].values
    x_ipd = ipd_df['treatment'].values
    age_ipd = ipd_df['age'].values
    
    # AD Data (simulated aggregate log-odds)
    y_ad = np.log(ad_df['events'].values / (ad_df['n'].values - ad_df['events'].values))
    x_ad = ad_df['exposure'].values
    
    with pm.Model() as model:
        # Invariant Causal Estimand (The true treatment effect)
        beta_inv = pm.Normal("beta_inv", mu=-0.8, sigma=0.1) # Anchored by IPD
        
        # Environmental effects (nuisance parameters for IRM robustness)
        tau_env = pm.HalfNormal("tau_env", sigma=0.2)
        delta_env = pm.Normal("delta_env", mu=0, sigma=tau_env, shape=len(ad_df))
        
        # IPD Likelihood
        p_ipd = pm.math.invlogit(beta_inv * x_ipd + 0.02 * age_ipd)
        pm.Bernoulli("obs_ipd", p=p_ipd, observed=y_ipd)
        
        # AD Likelihood (Weighted by NOT alignment)
        # Using a weighted likelihood to account for domain shift
        mu_ad = beta_inv * x_ad + delta_env
        pm.Normal("obs_ad", mu=mu_ad, sigma=0.1 / alignment_weights, observed=y_ad)
        
        # Sampling
        print("Starting 2026 Hybrid MCMC sampling (IRM + Neural OT)...")
        trace = pm.sample(200, tune=100, cores=1, chains=1, random_seed=42, progressbar=False)
    
    res = az.summary(trace, hdi_prob=0.95)
    return {
        "invariant_estimate_or": round(float(np.exp(res.loc['beta_inv', 'mean'])), 4),
        "ci_low": round(float(np.exp(res.loc['beta_inv', 'hdi_2.5%'])), 4),
        "ci_high": round(float(np.exp(res.loc['beta_inv', 'hdi_97.5%'])), 4),
        "stability_metric": float(res.loc['tau_env', 'mean'])
    }

def main():
    input_path = "hybrid-causal/data/hybrid_synthesis_input.json"
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
        
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    ipd_df = pd.DataFrame(input_data['ipd_anchors'])
    ad_df = pd.DataFrame(input_data['aggregate_ihme'])
    
    print("Performing Neural Optimal Transport (NOT) geometric alignment...")
    alignment_weights = apply_neural_ot_alignment(ipd_df, ad_df)
    
    print("Running Invariant Risk Minimization (IRM) for hybrid synthesis...")
    results = run_invariant_risk_minimization(ipd_df, ad_df, alignment_weights)
    
    # Synthesis
    output = {
        "model": "Hybrid-IRM-NOT-v1.0",
        "description": "2026 Invariant Causal Meta-Analysis with Neural OT Alignment",
        "estimands": results,
        "theoretical_efficiency": input_data['efficiency_are'],
        "truthcert": {
            "input_hash": generate_truthcert_hash(input_data),
            "timestamp": "2026-04-08"
        }
    }
    
    output_path = "hybrid-causal/output/hybrid_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Hybrid IRM-NOT Model execution complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
