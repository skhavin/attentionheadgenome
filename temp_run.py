
import sys
import importlib.util
sys.path.append('headgenome2_circuits')
spec = importlib.util.spec_from_file_location("profiler", "headgenome2_circuits/utils/head_profiler.py")
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)
profiler.profile_heads('llama-1b')
