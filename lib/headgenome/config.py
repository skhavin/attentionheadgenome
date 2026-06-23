"""
headgenome.config
─────────────────
Single source of truth for all HeadGenome compiler settings.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class HeadGenomeConfig:
    """Configuration for the HeadGenome compiler.

    Quick start
    -----------
    >>> from headgenome import HeadGenomeConfig
    >>> cfg = HeadGenomeConfig(mode="sparse_prefill", backend="flex", window=512)

    Parameters
    ----------
    mode : str
        One of:
          - "sparse_prefill"  — sliding-window masks during prefill (reduces TTFT)
          - "kv_eviction"     — head-granular KV eviction during decode (reduces TPS)
    backend : str
        One of:
          - "flex"   — FlexAttention + BlockMask (fastest, CUDA required)
          - "torch"  — Dense additive mask (-inf) via pre-hooks (correctness reference)
          - "dense"  — Full attention, no mask (baseline only)
    window : int
        Sliding-window size W for local/sink heads (default 512).
    preserve_roles : list[str]
        Head roles that receive full causal attention. Default ["retrieval", "induction"].
    retrieval_threshold : float
        δ threshold above which a head is labeled "retrieval". Default 0.30.
    induction_threshold : float
        δ threshold below which a head is labeled "induction". Default -0.50.
    sink_size : int
        Number of initial (sink) tokens always attended to in local heads. Default 4.
    compile : bool
        If True, torch.compile() is applied to the flex kernel. Default False.
    dry_run : bool
        If True, compile() prints the policy plan but does NOT patch the model.
    """

    mode: Literal["sparse_prefill", "kv_eviction"] = "sparse_prefill"
    backend: Literal["flex", "torch", "dense"] = "flex"
    window: int = 512
    preserve_roles: List[str] = field(default_factory=lambda: ["retrieval", "induction"])
    retrieval_threshold: float = 0.30
    induction_threshold: float = -0.50
    sink_threshold_entropy: float = 0.10
    sink_size: int = 4
    compile: bool = False
    dry_run: bool = False

    def __post_init__(self):
        valid_modes = {"sparse_prefill", "kv_eviction"}
        valid_backends = {"flex", "torch", "dense"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got '{self.backend}'")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    @classmethod
    def help(cls):
        """Print a quick-reference guide."""
        print("""
HeadGenomeConfig — Quick Reference
═══════════════════════════════════════════════════════════

  from headgenome import HeadGenomeConfig

  config = HeadGenomeConfig(
      mode="sparse_prefill",   # or "kv_eviction"
      backend="flex",          # or "torch" (correctness) or "dense" (baseline)
      window=512,              # sliding-window tokens for local/sink heads
      preserve_roles=[         # roles that keep full causal attention
          "retrieval",
          "induction",
      ],
      retrieval_threshold=0.30,   # δ cutoff for labeling retrieval heads
      induction_threshold=-0.50,  # δ cutoff for labeling induction heads
      sink_size=4,                # # of initial tokens always attended to
      compile=False,              # torch.compile() the flex kernel?
      dry_run=False,              # print plan without patching the model?
  )

  Then pass to HeadGenomeCompiler:

  from headgenome import HeadGenomeCompiler
  compiler = HeadGenomeCompiler("Qwen/Qwen2.5-1.5B", config=config)
  model = compiler.fit_compile_apply()
""")
