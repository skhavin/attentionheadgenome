import numpy as np
import scipy.stats as stats

def run_rsa_power(n_pairs, rho=0.59, alpha=0.05, iterations=5000):
    sig_count = 0
    for _ in range(iterations):
        cov = [[1.0, rho], [rho, 1.0]]
        data = np.random.multivariate_normal([0, 0], cov, size=n_pairs)
        x = data[:, 0]
        y = data[:, 1]
        
        corr, p_val = stats.spearmanr(x, y)
        if p_val < alpha and corr > 0:
            sig_count += 1
            
    return sig_count / iterations

def main():
    print("Power Analysis for RSA Spearman Correlation (rho=0.59)")
    req_k = None
    for k in range(5, 15):
        n_pairs = k * (k - 1) // 2
        power = run_rsa_power(n_pairs, rho=0.59)
        print(f"Categories: {k} -> Pairs: {n_pairs} -> Power: {power:.1%}")
        if power >= 0.8 and req_k is None:
            req_k = k
            break
            
    print(f">> Pre-registered requirement: We must test {req_k} computation categories.")

if __name__ == "__main__":
    main()
