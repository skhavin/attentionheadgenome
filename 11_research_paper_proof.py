import os
import math

def calculate_flops():
    print("="*80)
    print("   ATTENTION FLOP SCALING PROOF (THE O(N) COMPREHENSIVE CURVE)")
    print("="*80)
    
    # Model assumptions (based on a standard 1B-2B parameter model)
    D = 64  # Head dimension
    W = 256 # Local Window size
    S = 4   # Sink size
    E = 256 # Early Exit average search depth (empirical average chunk exits)

    # Context lengths from 1K to 1M
    seq_lengths = [1024, 4096, 16384, 32768, 131072, 1048576]
    
    print(f"\n[Assumptions]")
    print(f"Head Dim (D) = {D}")
    print(f"Local Window (W) = {W}")
    print(f"Sink Window (S) = {S}")
    print(f"Early Exit Avg Depth (E) = {E}")
    print(f"Universal Router Profile = 60% Local, 5% Sink, 35% Retrieval\n")
    
    print("| Context Length (N) | Baseline Dense FLOPs | Universal Router (Phase 1) | Universal + Early Exit (Phase 2) | Total Speedup |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for N in seq_lengths:
        # 1. Baseline Dense Attention FLOPs (per head)
        # Q*K^T = N x N dot products of length D = N^2 * D multiply-adds = 2 * N^2 * D FLOPs
        # Attn*V = N x N dot products of length D = N^2 * D multiply-adds = 2 * N^2 * D FLOPs
        # Total = 4 * N^2 * D
        dense_flops = 4 * (N**2) * D
        
        # 2. Universal Router (Phase 1) FLOPs
        local_flops = 4 * (N * W) * D if N > W else dense_flops
        sink_flops = 4 * (N * S) * D if N > S else dense_flops
        retrieval_dense = dense_flops
        
        phase1_flops = (0.60 * local_flops) + (0.05 * sink_flops) + (0.35 * retrieval_dense)
        
        # 3. Universal Router + Early Exit (Phase 2) FLOPs
        retrieval_early = 4 * (N * E) * D if N > E else dense_flops
        phase2_flops = (0.60 * local_flops) + (0.05 * sink_flops) + (0.35 * retrieval_early)
        
        def format_flops(f):
            if f > 1e12: return f"{f/1e12:.2f} TeraFLOPs"
            if f > 1e9: return f"{f/1e9:.2f} GigaFLOPs"
            if f > 1e6: return f"{f/1e6:.2f} MegaFLOPs"
            return f"{f:,.0f} FLOPs"
            
        reduction = dense_flops / phase2_flops
        
        print(f"| **{N:,}** | {format_flops(dense_flops)} | {format_flops(phase1_flops)} ({dense_flops/phase1_flops:.1f}x) | **{format_flops(phase2_flops)}** | **{reduction:.0f}x** |")

if __name__ == "__main__":
    calculate_flops()
