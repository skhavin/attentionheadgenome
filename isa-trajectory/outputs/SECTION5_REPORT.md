# Section 5: Generator Analysis (Direct Trajectory Attribution)

Having established in Sections 2 and 3 that the semantic trajectory (the F-Ratio manifold) is a real, conserved geometric structure, we now ask: *Who writes it?* Which components actively construct this geometry as the residual stream flows forward?

To answer this, we introduce **Dual-Metric Direct Trajectory Attribution (DTA)**. Unlike standard DLA which projects onto the static vocabulary logits, DTA projects component outputs onto the *evolving geometric trajectory* ($\Delta \hat{T}_L = \frac{C_{Arith} - C_{Sort}}{||\dots||}$). 

To isolate true geometric steerers from generic norm artifacts, we computed two metrics for every Attention Head and MLP across all 28 layers:
1. **Static Projection:** $DTA_{static} = \text{ComponentOutput} \cdot \Delta \hat{T}_L$. The absolute projection length onto the trajectory.
2. **Transition Contribution:** The percentage share a component contributes to the *actual* layer-to-layer geometric transition ($\Delta r_L \cdot \Delta \hat{T}_L$).

## Findings: MLPs Are The Sole Geometric Builders

The results present one of the cleanest functional isolations in this architecture. **Attention Heads contribute almost nothing to the structural branching of the semantic manifold; the geometry is constructed entirely by MLPs.**

During the critical "Rise Phase" of the manifold (Layers 10-20), where `Arithmetic` orthogonalizes from `Sorting`, the Transition DTA reveals massive MLP dominance:

| Component | Transition Share (Rise Phase) |
| :--- | :--- |
| **L16 MLP** | 106.10% |
| **L14 MLP** | 98.83% |
| **L10 MLP** | 95.82% |
| **L18 MLP** | 87.93% |
| **L19 MLP** | 83.29% |

Not a single Attention Head broke 25% positive transition share during the entire Rise Phase. In fact, when we analyzed the components with *negative* transition shares (the "Opponents" that push the residual stream away from the target trajectory, forcing the MLPs to overcompensate with $>100\%$ shares), **every single one of the top 10 opponents was an Attention Head**.

| Opponent Component | Transition Share (Rise Phase) |
| :--- | :--- |
| **L15 H5** | -20.67% |
| **L17 H3** | -15.21% |
| **L19 H6** | -14.79% |
| **L11 H8** | -13.40% |
| **L16 H6** | -12.60% |

This reveals a fascinating structural dynamic: the MLPs serve as the distributed semantic memory banks that actively push the residual stream outward along the macroscopic trajectory branches, while the Attention Heads consistently drag the state *backwards* (likely because they are attending to generic syntax or source tokens that dilute the task-specific semantic direction).

## The Norm Artifact Flag

The Dual-Metric approach successfully flagged several late-layer MLPs as norm artifacts. For example, the L26 MLP had a massive Static Projection of -14.625, but a Transition Contribution of exactly $0.00\%$. Without the transition metric, we would have incorrectly concluded that Layer 26 actively builds the trajectory, when in fact it simply outputs a large generic vector that happens to possess a static dot product with the trajectory direction.

## Cross-Pair Validation (Fact Recall vs Comparison)

To ensure "MLP Dominance" is an architectural universal and not an artifact of the Arithmetic vs Sorting manifold, we reran the exact same Dual-Metric DTA on a second, distinct category pair: `Fact Recall` vs `Comparison`.

The structural signature replicated perfectly. During the Rise Phase of the Fact Recall trajectory:
- **L10 MLP** provided 91.38% of the transition share.
- **L19 MLP** provided 80.33%.
- **L16 MLP** provided 64.69%.
- The top 4 positive transition components were all MLPs.
- The "Opponents" dragging the residual stream backward were once again dominated by Attention Heads (e.g., L19 H6 at -20.64%, L20 H7 at -16.68%).

**Conclusion:** The macroscopic geometric manifold is constructed sequentially and almost exclusively by mid-to-late layer MLPs. This structural division of labor (MLPs as semantic trajectory builders, Attention Heads as local syntactic routers or trajectory opponents) holds robustly across different semantic task categories.
