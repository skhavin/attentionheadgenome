"""headgenome.benchmarks package"""
from .speed import measure_ttft, measure_e2e, full_speed_benchmark
from .ppl   import measure_ppl
from .niah  import run_niah
