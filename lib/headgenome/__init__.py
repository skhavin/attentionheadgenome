"""
headgenome
──────────
HeadGenome: Attention Head Taxonomy for Efficient LLM Inference.

Quick start
───────────
  from headgenome import optimize
  model, report = optimize("Qwen/Qwen2.5-1.5B")

  from headgenome import HeadGenome
  HeadGenome.help()
"""

from .config   import HeadGenomeConfig
from .compiler import HeadGenome, optimize
from .taxonomy import TaxonomyResult, HeadLabel

__version__ = "0.1.0"
__all__ = ["HeadGenome", "HeadGenomeConfig", "TaxonomyResult", "HeadLabel", "optimize"]
