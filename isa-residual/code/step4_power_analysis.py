import numpy as np
import scipy.stats as stats
import json
import os

def monte_carlo_power(n_categories, effect_size, alpha=0.05, n_sims=1000):
    n_pairs = (n_categories * (n_categories - 1)) // 2
    significant_count = 0
    
    for _ in range(n_sims):
        # We simulate the true underlying similarity matrix vectors for Model A and Model B
        # Let Z be the true universal structure. 
        # A = Z + noise_A
        # B = Z + noise_B
        # We want corr(A, B) ≈ effect_size.
        
        # We can simulate this by drawing correlated Gaussians.
        cov_matrix = np.array([[1.0, effect_size], [effect_size, 1.0]])
        samples = np.random.multivariate_normal([0, 0], cov_matrix, size=n_pairs)
        
        A = samples[:, 0]
        B = samples[:, 1]
        
        corr, p_val = stats.pearsonr(A, B)
        # Using pearson as a proxy for the mantel test significance bound
        if p_val < alpha:
            significant_count += 1
            
    power = significant_count / n_sims
    return power

def main():
    observed_effect_size = 0.9644 # From the Step 0 deconfounding
    n_cat = 12
    n_sims = 10000
    
    print("Running Post-Hoc Descriptive Power Analysis...")
    print(f"Observed Deconfounded Effect Size (from N=8 pilot): {observed_effect_size}")
    print(f"Target Scale-up Categories: {n_cat}")
    
    power = monte_carlo_power(n_categories=n_cat, effect_size=observed_effect_size, n_sims=n_sims)
    
    print(f"Result: Given the effect size of {observed_effect_size}, N=12 categories provides {power * 100:.2f}% power at alpha=0.05.")
    
    os.makedirs("../outputs-isa-residual/step4_scaleup", exist_ok=True)
    with open("../outputs-isa-residual/step4_scaleup/power_analysis.json", "w") as f:
        json.dump({
            "observed_effect_size": observed_effect_size,
            "target_categories": n_cat,
            "power_percentage": float(power * 100)
        }, f, indent=2)

if __name__ == "__main__":
    main()
