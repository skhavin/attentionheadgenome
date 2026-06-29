"""
audit_head_vocabulary.py

Phase 9 - Lexical Audit: Run all 4 models over WikiText-103 validation sentences.
For every canonical attention head, aggregate which English tokens it attends to most.
Outputs:
  - outputs/phase9_semantics/vocab_audit_<Model>.json  (per-model data)
  - outputs/phase9_semantics/global_vocabulary_audit.html  (interactive table + visual)

Key fix: canonical_labels.json uses underscore keys "layer_head" (e.g. "4_13").
"""

import os
import json
import math
import torch
import numpy as np
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
MODELS = {
    "GPT-2":        "gpt2-medium",
    "Qwen-0.5B":    "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B":    "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR   = "outputs/phase9_semantics"
NUM_SEQS  = 60   # number of WikiText docs to process per model
SEQ_LEN   = 128  # max tokens per doc
TOP_N     = 5    # top-N tokens to store per head
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COLORS = {
    "Sink":      "#ef4444",  # red
    "Local":     "#22c55e",  # green
    "Retrieval": "#3b82f6",  # blue
    "Induction": "#f59e0b",  # amber
    "Unknown":   "#64748b",  # slate
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_token(tok_str: str) -> str:
    """Normalize subword tokens: strip Ġ (GPT-2) and ▁ (SentencePiece) prefixes."""
    t = tok_str.replace("Ġ", "").replace("▁", "").strip()
    return t if t else "[SPC]"

def load_canonical_labels() -> dict:
    with open("outputs/canonical_labels.json") as f:
        return json.load(f)

def get_label_map(labels_data: dict, model_name: str) -> dict:
    """Return {underscore_key: label_string} e.g. {'4_13': 'sink'}"""
    if model_name not in labels_data.get("models", {}):
        return {}
    return {k: v["label"] for k, v in labels_data["models"][model_name]["heads"].items()}

# ── Per-model audit ───────────────────────────────────────────────────────────
def run_audit(model_name: str, hf_id: str, texts: list, label_map: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Auditing: {model_name}  ({hf_id})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        attn_implementation="eager",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model.eval().to(DEVICE)

    # (layer, head) -> Counter of target tokens
    head_vocab: dict[tuple, Counter] = defaultdict(Counter)

    for text in tqdm(texts[:NUM_SEQS], desc="Sequences"):
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=SEQ_LEN, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        if seq_len < 10:
            continue

        input_ids = inputs["input_ids"][0].tolist()
        tokens = [clean_token(tokenizer.decode([tid])) for tid in input_ids]

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        for layer_idx, layer_attn in enumerate(out.attentions):
            attn = layer_attn[0].float().cpu().numpy()  # (heads, seq, seq)
            for head_idx in range(attn.shape[0]):
                mat = attn[head_idx]          # (seq, seq)
                top_keys = np.argmax(mat, axis=1)  # for each query row, top key
                for q_i, k_i in enumerate(top_keys):
                    if q_i == 0:
                        continue
                    head_vocab[(layer_idx, head_idx)][tokens[k_i]] += 1

    del model
    torch.cuda.empty_cache()

    # ── Format results ────────────────────────────────────────────────────────
    results = {}
    for (layer, head), counter in sorted(head_vocab.items()):
        key = f"{layer}_{head}"                          # ← underscore key
        label_raw = label_map.get(key, "unknown")
        label     = label_raw.capitalize()
        total     = sum(counter.values())

        top_tokens = [
            {"token": tok, "count": cnt,
             "percentage": round(100 * cnt / total, 1) if total else 0}
            for tok, cnt in counter.most_common(TOP_N)
        ]
        results[key] = {
            "layer":      layer,
            "head":       head,
            "label":      label,
            "color":      LABEL_COLORS.get(label, LABEL_COLORS["Unknown"]),
            "top_tokens": top_tokens,
        }

    # Save per-model JSON
    path = os.path.join(OUT_DIR, f"vocab_audit_{model_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → Saved {path}")
    return results


# ── HTML generator ────────────────────────────────────────────────────────────
def generate_html(all_results: dict[str, dict]):
    print("\nGenerating HTML...")

    # ── Build table rows ──────────────────────────────────────────────────────
    table_rows = ""
    chart_data_js = "const chartData = [\n"

    for model_name, results in all_results.items():
        for key, d in results.items():
            label = d["label"]
            color = d["color"]
            badge_cls = label.lower()
            toks_html = " ".join(
                f"<span class='tok'>{t['token']}"
                f"<em>{t['percentage']}%</em></span>"
                for t in d["top_tokens"]
            )
            table_rows += (
                f"<tr>"
                f"<td><b>{model_name}</b></td>"
                f"<td>{d['layer']}</td>"
                f"<td>{d['head']}</td>"
                f"<td><span class='badge badge-{badge_cls}'>{label}</span></td>"
                f"<td class='toks'>{toks_html}</td>"
                f"</tr>\n"
            )
            top1_pct = d["top_tokens"][0]["percentage"] if d["top_tokens"] else 0
            top1_tok = d["top_tokens"][0]["token"] if d["top_tokens"] else ""
            chart_data_js += (
                f"  {{ model: {json.dumps(model_name)}, layer: {d['layer']}, "
                f"head: {d['head']}, label: {json.dumps(label)}, "
                f"color: {json.dumps(color)}, top1pct: {top1_pct}, "
                f"top1tok: {json.dumps(top1_tok)} }},\n"
            )

    chart_data_js += "];\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HeadGenome — Vocabulary Audit</title>
<!-- DataTables -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<style>
  :root {{
    --bg:      #0b1120;
    --surface: #111827;
    --surface2:#1e293b;
    --border:  #334155;
    --sky:     #38bdf8;
    --text:    #f1f5f9;
    --muted:   #94a3b8;
    --sink:    #ef4444;
    --local:   #22c55e;
    --retrieval:#3b82f6;
    --induction:#f59e0b;
    --unknown: #64748b;
  }}
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; padding:32px; }}
  h1 {{ text-align:center; font-size:2.4rem; color:var(--sky);
       background:linear-gradient(135deg,#38bdf8,#818cf8);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px; }}
  .subtitle {{ text-align:center; color:var(--muted); max-width:780px; margin:0 auto 48px; line-height:1.7; }}

  /* ── Legend ── */
  .legend {{ display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin-bottom:36px; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; font-size:.9rem; }}
  .legend-dot {{ width:14px; height:14px; border-radius:50%; }}

  /* ── Scatter ── */
  #scatter-wrap {{ background:var(--surface); border-radius:16px;
                   padding:24px; margin-bottom:48px;
                   box-shadow:0 8px 32px rgba(0,0,0,.5); }}
  #scatter-wrap h2 {{ color:var(--sky); margin-bottom:16px; font-size:1.3rem; }}
  #scatter-canvas {{ display:block; width:100%; border-radius:8px; }}

  /* ── Tooltip ── */
  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(15,23,42,.95); border:1px solid var(--border);
    border-radius:8px; padding:10px 14px; font-size:.85rem;
    box-shadow:0 4px 16px rgba(0,0,0,.6); max-width:260px; z-index:9999;
  }}
  #tooltip strong {{ color:var(--sky); display:block; margin-bottom:4px; }}
  #tooltip .ttok {{ display:inline-block; background:var(--surface2);
                    border:1px solid var(--border); border-radius:4px;
                    padding:1px 5px; margin:2px; font-family:monospace; font-size:.8rem; }}

  /* ── Table wrapper ── */
  .table-wrap {{ background:var(--surface); border-radius:16px;
                 padding:28px; box-shadow:0 8px 32px rgba(0,0,0,.5); }}
  .table-wrap h2 {{ color:var(--sky); margin-bottom:20px; font-size:1.3rem; }}

  table.dataTable {{ color:var(--text) !important; border-collapse:collapse !important; width:100% !important; }}
  table.dataTable thead th {{ background:var(--surface2); color:var(--sky);
                               border-bottom:2px solid var(--sky) !important; padding:10px; }}
  table.dataTable tbody tr {{ background:var(--surface) !important; border-bottom:1px solid var(--border); }}
  table.dataTable tbody tr:nth-child(even) {{ background:var(--bg) !important; }}
  table.dataTable tbody tr:hover td {{ background:var(--surface2) !important; }}
  td {{ padding:8px 10px !important; vertical-align:top; }}

  .badge {{ display:inline-block; padding:3px 9px; border-radius:20px;
            font-size:.78rem; font-weight:700; letter-spacing:.04em; }}
  .badge-sink       {{ background:var(--sink);      color:#fff; }}
  .badge-local      {{ background:var(--local);     color:#000; }}
  .badge-retrieval  {{ background:var(--retrieval); color:#fff; }}
  .badge-induction  {{ background:var(--induction); color:#000; }}
  .badge-unknown    {{ background:var(--unknown);   color:#fff; }}

  .toks {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .tok {{ background:var(--surface2); border:1px solid var(--border);
          border-radius:5px; padding:2px 7px; font-family:monospace; font-size:.8rem;
          display:flex; gap:4px; align-items:center; }}
  .tok em {{ color:#fca5a5; font-style:normal; font-size:.75rem; }}

  /* DT overrides */
  .dataTables_wrapper .dataTables_length,
  .dataTables_wrapper .dataTables_filter,
  .dataTables_wrapper .dataTables_info,
  .dataTables_wrapper .dataTables_paginate {{ color:var(--muted); margin-top:10px; }}
  .dataTables_wrapper .dataTables_filter input {{
    background:var(--surface2); border:1px solid var(--border);
    color:var(--text); border-radius:6px; padding:4px 10px; outline:none;
  }}
  .dataTables_wrapper .dataTables_paginate .paginate_button {{
    color:var(--muted) !important; border-radius:4px; border:none !important; }}
  .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
    background:var(--sky) !important; color:#0b1120 !important; border:none !important; }}
  .dataTables_wrapper .dataTables_paginate .paginate_button:hover {{
    background:var(--surface2) !important; color:var(--text) !important; }}
</style>
</head>
<body>

<h1>HeadGenome — Global Vocabulary Audit</h1>
<p class="subtitle">
  Across <strong>all 4 architectures</strong>, every attention head is mapped to its
  real English lexical targets extracted from <strong>WikiText-103</strong>.
  Hover over any dot in the scatter to see which words that head actually "stares at" in natural text.
  The table below is fully searchable and sortable.
</p>

<!-- Legend -->
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> Sink</div>
  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Local</div>
  <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div> Retrieval</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div> Induction</div>
  <div class="legend-item"><div class="legend-dot" style="background:#64748b"></div> Unknown</div>
</div>

<!-- Scatter -->
<div id="scatter-wrap">
  <h2>📍 Head Anatomy: Layer × Head — color = classification, size = top-1 lexical dominance (%)</h2>
  <canvas id="scatter-canvas"></canvas>
</div>

<div id="tooltip"></div>

<!-- Table -->
<div class="table-wrap">
  <h2>🔬 Lexical Audit Table — Top 5 vocabulary targets per head</h2>
  <table id="auditTable">
    <thead>
      <tr>
        <th>Model</th><th>Layer</th><th>Head</th>
        <th>Classification</th><th>Top Lexical Targets</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</div>

<script>
{chart_data_js}

// ── Scatter ──────────────────────────────────────────────────────────────────
const canvas = document.getElementById('scatter-canvas');
const ctx    = canvas.getContext('2d');
const tip    = document.getElementById('tooltip');

// Group by model
const models = [...new Set(chartData.map(d=>d.model))];
const MODEL_Y_BANDS = {{}};
models.forEach((m,i) => {{ MODEL_Y_BANDS[m] = i; }});

const DPR    = window.devicePixelRatio || 1;
const PAD    = {{ top:40, right:30, bottom:60, left:180 }};

function drawScatter() {{
  const W = canvas.parentElement.clientWidth - 4;
  const H = Math.max(models.length * 180, 500);
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  canvas.width  = W * DPR;
  canvas.height = H * DPR;
  ctx.scale(DPR, DPR);

  const plotW = W - PAD.left - PAD.right;
  const bandH = (H - PAD.top - PAD.bottom) / models.length;

  // Background
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W, H);

  // Grid lines & model labels
  models.forEach((m, mi) => {{
    const bandTop = PAD.top + mi * bandH;
    const bandCy  = bandTop + bandH / 2;

    // band bg alternating
    ctx.fillStyle = mi % 2 === 0 ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0)';
    ctx.fillRect(PAD.left, bandTop, plotW, bandH);

    // model label
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 14px Segoe UI';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(m, PAD.left - 12, bandCy);

    // horizontal midline
    ctx.strokeStyle = 'rgba(148,163,184,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, bandCy);
    ctx.lineTo(PAD.left + plotW, bandCy);
    ctx.stroke();
  }});

  // Compute max layer & head for axis
  const maxLayer = Math.max(...chartData.map(d=>d.layer));
  const maxHead  = Math.max(...chartData.map(d=>d.head));

  // X axis label
  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px Segoe UI';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Layer Index →', PAD.left + plotW/2, H - PAD.bottom + 28);

  // Vertical grid
  for (let l = 0; l <= maxLayer; l += 4) {{
    const x = PAD.left + (l / maxLayer) * plotW;
    ctx.strokeStyle = 'rgba(148,163,184,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, PAD.top);
    ctx.lineTo(x, H - PAD.bottom);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Segoe UI';
    ctx.textBaseline = 'top';
    ctx.fillText(l, x, H - PAD.bottom + 6);
  }}

  // Dots
  const dots = [];
  chartData.forEach(d => {{
    const mi   = MODEL_Y_BANDS[d.model];
    const bandTop = PAD.top + mi * bandH;
    const bandH_  = bandH;
    // jitter on head axis (vertical within band)
    const jitter = ((d.head / Math.max(maxHead,1)) - 0.5) * bandH_ * 0.72;
    const cx = PAD.left + (d.layer / Math.max(maxLayer,1)) * plotW;
    const cy = bandTop + bandH_/2 + jitter;
    const r  = 4 + (d.top1pct / 100) * 10;  // size encodes dominance
    dots.push({{ ...d, cx, cy, r }});
  }});

  // draw in label order: local first (background), then specialized on top
  const ORDER = ['Local','Unknown','Sink','Retrieval','Induction'];
  ORDER.forEach(lbl => {{
    dots.filter(d=>d.label===lbl).forEach(d => {{
      ctx.beginPath();
      ctx.arc(d.cx, d.cy, d.r, 0, Math.PI*2);
      ctx.fillStyle = d.color + (lbl==='Local'?'88':'cc');
      ctx.fill();
      if (lbl !== 'Local') {{
        ctx.strokeStyle = d.color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }}
    }});
  }});

  return dots;
}}

let dots = drawScatter();

// Tooltip on mousemove
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const hit = dots.find(d => Math.hypot(d.cx-mx, d.cy-my) < d.r + 4);
  if (hit) {{
    const tokHtml = hit.top1tok
      ? `<div style="margin-top:6px">Top target: <span class="ttok">${{hit.top1tok}}</span> (top-1: ${{hit.top1pct}}%)</div>`
      : '';
    tip.innerHTML = `<strong>${{hit.model}} — L${{hit.layer}} H${{hit.head}}</strong>
      <span style="background:${{hit.color}};color:#000;border-radius:4px;padding:1px 7px;font-size:.78rem">${{hit.label}}</span>
      ${{tokHtml}}`;
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top  = (e.clientY - 10) + 'px';
    canvas.style.cursor = 'pointer';
  }} else {{
    tip.style.display = 'none';
    canvas.style.cursor = 'default';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tip.style.display='none'; }});
window.addEventListener('resize', () => {{ dots = drawScatter(); }});

// ── DataTable ────────────────────────────────────────────────────────────────
$(document).ready(function() {{
  $('#auditTable').DataTable({{
    pageLength: 30,
    order: [[0,'asc'],[1,'asc'],[2,'asc']],
    columnDefs: [{{ targets:[4], orderable:false }}]
  }});
}});
</script>
</body>
</html>
"""

    out_path = os.path.join(OUT_DIR, "global_vocabulary_audit.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → Generated: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading canonical labels …")
    labels_data = load_canonical_labels()

    print("Loading WikiText-103 validation split …")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.split()) > 30]
    print(f"  {len(texts)} usable documents found.")

    all_results = {}
    for model_name, hf_id in MODELS.items():
        label_map = get_label_map(labels_data, model_name)
        results   = run_audit(model_name, hf_id, texts, label_map)
        all_results[model_name] = results

    generate_html(all_results)
    print("\nAll done! Open outputs/phase9_semantics/global_vocabulary_audit.html")


if __name__ == "__main__":
    main()
