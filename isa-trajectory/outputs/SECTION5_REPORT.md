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

Not a single Attention Head broke 25% transition share during the entire Rise Phase (the maximum was L15 H5 at -20.6%). The Attention Heads act as passive routers or local syntactic movers, while the MLPs serve as the distributed semantic memory banks that actively push the residual stream outward along the macroscopic trajectory branches. 

## The Norm Artifact Flag

The Dual-Metric approach successfully flagged several late-layer MLPs as norm artifacts. For example, the L26 MLP had a massive Static Projection of -14.625, but a Transition Contribution of exactly $0.00\%$. Without the transition metric, we would have incorrectly concluded that Layer 26 actively builds the trajectory, when in fact it simply outputs a large generic vector that happens to possess a static dot product with the trajectory direction.

**Conclusion:** The macroscopic geometric manifold is constructed sequentially and almost exclusively by mid-to-late layer MLPs.
