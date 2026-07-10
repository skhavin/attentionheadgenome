import numpy as np
import scipy.stats as stats

def run_power_analysis(n_target, n_control, d=0.8, alpha=0.05, iterations=10000):
    sig_count = 0
    for _ in range(iterations):
        target_group = np.random.normal(loc=d, scale=1.0, size=n_target)
        control_group = np.random.normal(loc=0.0, scale=1.0, size=n_control)
        try:
            _, p_val = stats.mannwhitneyu(target_group, control_group, alternative='greater')
            if p_val < alpha:
                sig_count += 1
        except ValueError:
            pass
    return sig_count / iterations

def main():
    print("--- Phase 13 Monte Carlo Power Analysis ---")
    print("Effect size: Cohen's d = 0.8 (Moderate-to-large)")
    
    # Check our current Confirmation N per category (approx N=7)
    power_7 = run_power_analysis(7, 7*4, d=0.8)
    print(f"\nPower at current N=7 per category: {power_7:.1%}")
    
    print("\nDetermining required N per category for 80% power:")
    req_n = None
    for n in range(5, 50):
        # We test one category against the 4 other categories (N * 4 control)
        power = run_power_analysis(n, n*4, d=0.8, iterations=2000)
        if power >= 0.8:
            req_n = n
            print(f"N={n} (target) vs {n*4} (control) -> Power: {power:.1%}")
            break
            
    print(f"\n>> DECISION: We must generate {req_n} Confirmation prompts per category.")
    print(f">> Assuming 5 categories, we need a Discovery set of {req_n*2*5} prompts and a Confirmation set of {req_n*5} prompts.")

if __name__ == "__main__":
    main()
