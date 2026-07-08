"""
HeadGenome: Make your LLMs fast without fine-tuning.
Extremely simple O(N) universal attention compilation.
"""

from .compiler import compile
from .taxonomy import extract_head_taxonomy

__version__ = "1.0.0"
__all__ = ["compile", "extract_head_taxonomy"]
