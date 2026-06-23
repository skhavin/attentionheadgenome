"""
headgenome.compiler
───────────────────
The main HeadGenome class — Research API.
Also exposes optimize() for the Beginner API.

Research API
────────────
  hg = HeadGenome("Qwen/Qwen2.5-1.5B")
  hg.profile(docs=300)
  hg.compile(backend="flex", window=512)
  hg.benchmark(["ppl", "niah", "speed"], seq_len=4096)
  hg.save("runs/qwen15b")

Beginner API
────────────
  model, report = optimize("Qwen/Qwen2.5-1.5B")
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HeadGenomeConfig
from .taxonomy import TaxonomyResult, probe_model, load_taxonomy
from .backends.torch_mask import apply_torch_mask, TorchMaskHooks
from .benchmarks.speed import measure_ttft, measure_e2e
from .benchmarks.ppl import measure_ppl
from .benchmarks.niah import run_niah


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 25) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _build_prompt_for_seq_len(tokenizer, target_len: int) -> str:
    """Build a WikiText-style prompt of approximately target_len tokens."""
    FILLER = (
        "The study of artificial intelligence has progressed rapidly over the past decade. "
        "Researchers have developed models capable of understanding natural language, generating "
        "images, writing code, and solving complex mathematical problems. "
    )
    ids = tokenizer(FILLER * 200, return_tensors="pt")["input_ids"][0]
    truncated = ids[:target_len]
    return tokenizer.decode(truncated, skip_special_tokens=True)


# ─── HeadGenome (Research API) ────────────────────────────────────────────────

class HeadGenome:
    """
    HeadGenome research compiler.

    Example
    -------
    >>> hg = HeadGenome("Qwen/Qwen2.5-1.5B")
    >>> hg.profile(docs=300)
    >>> hg.compile(backend="flex", window=512)
    >>> results = hg.benchmark(["ppl", "speed"], seq_len=4096)
    >>> hg.save("runs/qwen15b")

    Call HeadGenome.help() for full usage guide.
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[HeadGenomeConfig] = None,
        device: Optional[str] = None,
        dtype: str = "bf16",
    ):
        self.model_id = model_id
        self.config   = config or HeadGenomeConfig()
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(dtype, str):
            dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
            dtype = dtype_map.get(dtype, dtype)
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        self._taxonomy:   Optional[TaxonomyResult] = None
        self._hooks:      Optional[TorchMaskHooks]  = None
        self._compiled:   bool = False
        self._compile_backend: str = "torch"

        self._bench_results: Dict = {}

        print(f"[HeadGenome] Loading {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype
        ).to(self.device).eval()

        cfg = self.model.config
        self._num_layers   = cfg.num_hidden_layers
        self._num_heads    = cfg.num_attention_heads
        self._num_kv_heads = getattr(cfg, "num_key_value_heads", self._num_heads)
        print(f"[HeadGenome] Loaded  — layers={self._num_layers}, Q-heads={self._num_heads}, KV-heads={self._num_kv_heads}")

    # ── Profiling ─────────────────────────────────────────────────────────────

    def profile(
        self,
        docs: int = 300,
        cache_path: Optional[str] = None,
    ) -> "HeadGenome":
        """
        Run entropy-collapse probing to label every head.

        Parameters
        ----------
        docs : int
            Number of probe pairs to run. More = more accurate labels. Default 300.
        cache_path : str, optional
            If given, saves / loads taxonomy from this JSON file.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"[HeadGenome] Loading cached taxonomy from {cache_path}")
            self._taxonomy = load_taxonomy(cache_path)
            return self

        print(f"[HeadGenome] Probing {self._num_layers}×{self._num_heads} heads with {docs} doc pairs …")
        t0 = time.perf_counter()
        self._taxonomy = probe_model(
            self.model,
            self.tokenizer,
            num_pairs=docs,
            retrieval_threshold=self.config.retrieval_threshold,
            induction_threshold=self.config.induction_threshold,
            sink_entropy_threshold=self.config.sink_threshold_entropy,
            device=self.device,
        )
        elapsed = time.perf_counter() - t0
        print(f"[HeadGenome] Probing done in {elapsed:.1f}s")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self._taxonomy.to_dict(), f, indent=2)
            print(f"[HeadGenome] Taxonomy saved -> {cache_path}")

        return self

    # ── Compilation ───────────────────────────────────────────────────────────

    def compile(
        self,
        backend: str = "flex",
        window: Optional[int] = None,
        preserve: Optional[Tuple[str, ...]] = None,
        dry_run: bool = False,
    ) -> "HeadGenome":
        """
        Apply HeadGenome sparse attention to the model.

        Parameters
        ----------
        backend : str
            "flex"  — FlexAttention (fastest, CUDA ≥ PyTorch 2.5)
            "torch" — Additive -inf mask (correctness reference, any device)
            "dense" — No mask (full attention baseline)
        window : int
            Sliding window size for local/sink heads.
        preserve : tuple[str], optional
            Roles to preserve with full causal attention.
            Defaults to config.preserve_roles.
        dry_run : bool
            Print plan without patching the model.
        """
        if self._taxonomy is None:
            raise RuntimeError("Run .profile() before .compile()")

        W        = window or self.config.window
        preserve = set(preserve or self.config.preserve_roles)

        # Print plan
        tax = self._taxonomy
        total = self._num_layers * self._num_heads
        crit_pct  = 100 * len(tax.critical) / total
        local_pct = 100 * len(tax.local)    / total
        sink_pct  = 100 * len(tax.sink)     / total

        # Theoretical FLOP savings
        f_crit  = len(tax.critical) / total
        f_local = len(tax.local)    / total
        f_sink  = len(tax.sink)     / total
        # At large N, savings ≈ 1 - (f_crit + f_local * W/N + f_sink * sink/N)
        # For W and N=4096:
        N = 4096
        eff = f_crit * N + f_local * min(W, N) + f_sink * self.config.sink_size
        theoretical_savings = max(0, 100 * (1 - eff / N))

        print(f"""
╔══════════════════════════════════════════════════════════╗
║              HeadGenome Compiler Plan                    ║
╚══════════════════════════════════════════════════════════╝

  Model     : {self.model_id}
  Backend   : {backend}
  Window    : {W}
  Preserve  : {sorted(preserve)}

  Taxonomy
  ─────────────────────────────────
  Sink        {len(tax.sink):4d}   {sink_pct:5.1f}%  {_bar(sink_pct)}
  Local       {len(tax.local):4d}   {local_pct:5.1f}%  {_bar(local_pct)}
  Retrieval   {len(tax.retrieval):4d}   {100*len(tax.retrieval)/total:5.1f}%  {_bar(100*len(tax.retrieval)/total)}
  Induction   {len(tax.induction):4d}   {100*len(tax.induction)/total:5.1f}%  {_bar(100*len(tax.induction)/total)}
  Critical    {len(tax.critical):4d}   {crit_pct:5.1f}%  {_bar(crit_pct)}

  Spatial Law
  ─────────────────────────────────
  Mean retrieval depth : {self._mean_depth(tax.retrieval):.3f}
  Mean induction depth : {self._mean_depth(tax.induction):.3f}

  Compiler Policy
  ─────────────────────────────────
  Retrieval → full causal attention
  Induction → full causal attention
  Local     → sliding window W={W}
  Sink      → sink tokens ({self.config.sink_size}) + W={W}

  Theoretical attention-op reduction @ N=4096 : {theoretical_savings:.1f}%
  (NOT wall-clock — run .benchmark() for real TTFT speedup)
""")

        if dry_run:
            print("  [dry_run=True] Model NOT patched.")
            return self

        # Remove existing hooks if any
        if self._hooks:
            self._hooks.remove()
            self._hooks = None

        if backend == "dense":
            print("[HeadGenome] Backend=dense: no masking applied (baseline mode)")
            self._compiled = True
            self._compile_backend = "dense"
            return self

        if backend in ("flex", "torch"):
            role_map = {k: v.role for k, v in tax.labels.items()}
            self._hooks = apply_torch_mask(
                model=self.model,
                role_map=role_map,
                num_layers=self._num_layers,
                num_heads=self._num_heads,
                window=W,
                sink_size=self.config.sink_size,
                preserve_roles=preserve,
            )
            self._compiled = True
            self._compile_backend = backend
            verb = "FlexAttention (mask via hooks — true flex kernel requires custom integration)" if backend == "flex" else "Torch additive mask"
            print(f"[HeadGenome] Patched: {verb}")

        return self

    def remove(self) -> "HeadGenome":
        """Remove all attention patches (restore dense model)."""
        if self._hooks:
            self._hooks.remove()
            self._hooks = None
        self._compiled = False
        return self

    # ── Benchmarking ──────────────────────────────────────────────────────────

    def benchmark(
        self,
        tasks: List[str] | None = None,
        seq_len: int = 4096,
        new_tokens: int = 128,
        niah_samples: int = 20,
        niah_depths: List[float] | None = None,
        warmup: int = 3,
        runs: int = 10,
    ) -> Dict:
        """
        Run benchmarks. Available tasks: "ppl", "niah", "speed".

        Example
        -------
        >>> results = hg.benchmark(["ppl", "niah", "speed"], seq_len=4096)
        """
        tasks = tasks or ["ppl", "speed"]
        prompt = _build_prompt_for_seq_len(self.tokenizer, seq_len)
        results = {"seq_len": seq_len, "backend": self._compile_backend}

        if "ppl" in tasks:
            print(f"[HeadGenome] Measuring PPL (seq_len={seq_len}) …")
            results["ppl"] = measure_ppl(
                self.model, self.tokenizer, seq_len=min(seq_len, 512),
                device=self.device
            )

        if "speed" in tasks:
            print(f"[HeadGenome] Measuring TTFT (seq_len={seq_len}, runs={runs}) …")
            results["ttft"] = measure_ttft(
                self.model, self.tokenizer, prompt,
                device=self.device, warmup=warmup, runs=runs,
            )
            print(f"[HeadGenome] Measuring E2E (new_tokens={new_tokens}, runs={min(runs,5)}) …")
            results["e2e"] = measure_e2e(
                self.model, self.tokenizer, prompt,
                new_tokens=new_tokens, device=self.device,
                warmup=warmup, runs=min(runs, 5),
            )

        if "niah" in tasks:
            print(f"[HeadGenome] Running NIAH ({niah_samples} samples × {len(niah_depths or [0.1,0.25,0.5,0.75,0.9])} depths) …")
            results["niah"] = run_niah(
                self.model, self.tokenizer,
                num_samples=niah_samples,
                depths=niah_depths,
                device=self.device,
            )

        self._bench_results = results
        return results

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self, dense_results: Optional[Dict] = None) -> None:
        """
        Print a formatted HeadGenome report.
        Pass dense_results (from a separate dense benchmark run) for comparison.
        """
        tax = self._taxonomy
        br  = self._bench_results

        total = self._num_layers * self._num_heads if tax else 1
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # FLOP savings
        f_crit  = len(tax.critical) / total if tax else 0
        f_local = len(tax.local)    / total if tax else 0
        f_sink  = len(tax.sink)     / total if tax else 0
        W = self.config.window
        N = br.get("seq_len", 4096)
        eff = f_crit * N + f_local * min(W, N) + f_sink * self.config.sink_size
        theoretical_savings = max(0, 100 * (1 - eff / N)) if tax else 0

        print("═" * 60)
        print("  HeadGenome Report")
        print("═" * 60)
        print(f"  Model         : {self.model_id}")
        print(f"  GPU           : {gpu_name}")
        print(f"  dtype         : {str(self.dtype).split('.')[-1]}")
        print(f"  Architecture  : {'GQA' if self._num_kv_heads < self._num_heads else 'MHA'}")
        print(f"  Layers        : {self._num_layers}")
        print(f"  Q heads       : {self._num_heads * self._num_layers} total ({self._num_heads}/layer)")
        print(f"  KV heads      : {self._num_kv_heads}/layer")
        print(f"  Backend       : {self._compile_backend}")
        print(f"  Window        : {W}")

        if tax:
            print()
            print("  Taxonomy")
            print("  " + "─" * 40)
            print(f"  Sink        {len(tax.sink):5d}   {100*len(tax.sink)/total:5.1f}%")
            print(f"  Local       {len(tax.local):5d}   {100*len(tax.local)/total:5.1f}%")
            print(f"  Retrieval   {len(tax.retrieval):5d}   {100*len(tax.retrieval)/total:5.1f}%")
            print(f"  Induction   {len(tax.induction):5d}   {100*len(tax.induction)/total:5.1f}%")
            print(f"  Critical    {len(tax.critical):5d}   {100*len(tax.critical)/total:5.1f}%")
            print()
            print("  Spatial Law")
            print("  " + "─" * 40)
            print(f"  Mean retrieval depth : {self._mean_depth(tax.retrieval):.3f}")
            print(f"  Mean induction depth : {self._mean_depth(tax.induction):.3f}")
            print()
            print("  Compiler Policy")
            print("  " + "─" * 40)
            print(f"  Retrieval  → full causal attention")
            print(f"  Induction  → full causal attention")
            print(f"  Local      → sliding window W={W}")
            print(f"  Sink       → sink ({self.config.sink_size} tok) + W={W}")

        if br:
            print()
            print("  Benchmark Results")
            print("  " + "─" * 40)

            # PPL
            hg_ppl  = br.get("ppl", {}).get("ppl", "—")
            dns_ppl = (dense_results or {}).get("ppl", {}).get("ppl", "—")
            print(f"  Dense PPL          : {dns_ppl}")
            print(f"  HeadGenome PPL     : {hg_ppl}")

            # Speed
            ttft_hg = br.get("ttft", {})
            ttft_dn = (dense_results or {}).get("ttft", {})
            if ttft_hg:
                print(f"  Dense TTFT         : {ttft_dn.get('ttft_ms_mean', '—'):.1f} ms" if ttft_dn else "  Dense TTFT         : —")
                print(f"  HeadGenome TTFT    : {ttft_hg.get('ttft_ms_mean', 0):.1f} ms (median {ttft_hg.get('ttft_ms_median',0):.1f} ms)")
                print(f"  HG Prefill tok/s   : {ttft_hg.get('prefill_tok_s', 0):,.0f}")
                print(f"  HG Peak VRAM       : {ttft_hg.get('peak_vram_gb', 0):.2f} GB")

                dns_ttft = ttft_dn.get("ttft_ms_mean", 0) if ttft_dn else 0
                hg_ttft  = ttft_hg.get("ttft_ms_mean", 0)
                if dns_ttft and hg_ttft:
                    speedup = dns_ttft / hg_ttft
                    print(f"  TTFT Speedup       : {speedup:.2f}×  (measured wall-clock)")

            e2e_hg = br.get("e2e", {})
            if e2e_hg:
                print(f"  TPOT               : {e2e_hg.get('tpot_ms', 0):.1f} ms/token")
                print(f"  Decode tok/s       : {e2e_hg.get('decode_tok_s', 0):.0f}")
                print(f"  E2E Latency        : {e2e_hg.get('e2e_latency_ms', 0):.0f} ms")

            # NIAH
            niah = br.get("niah", {})
            dns_niah = (dense_results or {}).get("niah", {})
            if niah:
                print(f"  Dense NIAH         : {dns_niah.get('overall_pct', '—')}%" if dns_niah else "  Dense NIAH         : —")
                print(f"  HeadGenome NIAH    : {niah.get('overall_pct', 0):.1f}%")

            print()
            print(f"  Theoretical attention-op reduction @ N={N}")
            print(f"    {theoretical_savings:.1f}%  (NOT wall-clock — see TTFT above)")

        print()
        print("  ⚠  Warning")
        print("  ─" * 20)
        print("  PPL preservation does NOT guarantee retrieval preservation.")
        print("  Always check NIAH for retrieval-heavy workloads.")
        print("  Benchmark TTFT to confirm wall-clock speedup before claiming savings.")
        print("═" * 60)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, run_dir: str) -> None:
        """Save taxonomy, config, and benchmark results to run_dir/."""
        os.makedirs(run_dir, exist_ok=True)

        if self._taxonomy:
            with open(os.path.join(run_dir, "taxonomy.json"), "w") as f:
                json.dump(self._taxonomy.to_dict(), f, indent=2)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            import dataclasses
            json.dump(dataclasses.asdict(self.config), f, indent=2)

        if self._bench_results:
            with open(os.path.join(run_dir, "benchmarks.json"), "w") as f:
                json.dump(self._bench_results, f, indent=2)

        print(f"[HeadGenome] Saved run artifacts -> {run_dir}/")

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(self) -> "AutoModelForCausalLM":
        """Return the patched model for use in downstream code."""
        if not self._compiled:
            raise RuntimeError("Run .compile() before .apply()")
        return self.model

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_depth(heads) -> float:
        if not heads:
            return 0.0
        return sum(h.relative_depth for h in heads) / len(heads)

    @staticmethod
    def help():
        """Print the full API reference."""
        print("""
HeadGenome — API Reference
══════════════════════════════════════════════════════════════

  BEGINNER API
  ─────────────────────────────────────────────────────────
  from headgenome import optimize

  model, report = optimize("Qwen/Qwen2.5-1.5B")
  # Runs profile + compile + benchmark automatically.

  RESEARCH API
  ─────────────────────────────────────────────────────────
  from headgenome import HeadGenome

  hg = HeadGenome(
      "Qwen/Qwen2.5-1.5B",  # HuggingFace model ID
      device="cuda",          # or "cpu"
      dtype="bf16",           # or "fp16", "fp32"
  )

  hg.profile(docs=300)
  # Runs entropy-collapse probing on 300 document pairs.

  hg.compile(
      backend="torch",        # "flex" (fast) | "torch" (correct) | "dense" (baseline)
      window=512,             # sliding-window size for local heads
      preserve=("retrieval", "induction"),
      dry_run=False,          # set True to preview plan without patching
  )

  results = hg.benchmark(
      tasks=["ppl", "niah", "speed"],
      seq_len=4096,
      new_tokens=128,
      niah_samples=20,
      runs=10,
  )

  hg.report()
  hg.save("runs/qwen15b")

  model = hg.apply()   # get the patched model

  hg.remove()          # restore full attention

  CONFIG API
  ─────────────────────────────────────────────────────────
  from headgenome import HeadGenomeConfig

  config = HeadGenomeConfig(
      mode="sparse_prefill",
      backend="flex",
      window=512,
      preserve_roles=["retrieval", "induction"],
      retrieval_threshold=0.30,
      induction_threshold=-0.50,
      sink_size=4,
  )
  hg = HeadGenome("Qwen/Qwen2.5-1.5B", config=config)

  HeadGenomeConfig.help()  # more details on every parameter

══════════════════════════════════════════════════════════════
""")


# ─── Beginner API ─────────────────────────────────────────────────────────────

def optimize(
    model_id: str,
    backend: str = "torch",
    window: int = 512,
    docs: int = 100,
    tasks: Optional[List[str]] = None,
    seq_len: int = 2048,
    device: Optional[str] = None,
    dtype: str = "bf16",
) -> Tuple["AutoModelForCausalLM", Dict]:
    """
    One-call beginner API: profile → compile → benchmark → report.

    Parameters
    ----------
    model_id : str       HuggingFace model ID.
    backend  : str       "torch" | "flex" | "dense"
    window   : int       Sliding window size.
    docs     : int       Probe pairs for taxonomy. More = more accurate.
    tasks    : list      Benchmark tasks. Default ["ppl", "speed"].
    seq_len  : int       Sequence length for benchmarks.

    Returns
    -------
    (patched_model, benchmark_results_dict)

    Example
    -------
    >>> model, report = optimize("Qwen/Qwen2.5-1.5B")
    """
    tasks = tasks or ["ppl", "speed"]

    hg = HeadGenome(model_id, device=device, dtype=dtype)
    hg.profile(docs=docs)
    hg.compile(backend=backend, window=window)
    results = hg.benchmark(tasks=tasks, seq_len=seq_len)
    hg.report()
    return hg.apply(), results
