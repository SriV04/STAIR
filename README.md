# STAIR — Scheduling- and Transport-Aware Intermediate Representation

STAIR (the code in this `CMIR` repository) is a research toolchain for exploring
**time-multiplexed ("folded") FPGA implementations** of the neural networks used
for **jet tagging** at the LHC — primarily the
[**JEDI-linear**](https://arxiv.org/abs/2508.15468) graph neural network, plus
Deep Sets and Linformer variants.

It takes a trained, quantized Keras model and:

1. lifts it into a hardware-oriented graph IR (**NN-IR**),
2. lowers that into schedulable primitives (**Sched-IR**),
3. **folds** the primitives (reuses a small number of compute lanes across many
   work units instead of building one lane per unit),
4. estimates the FPGA cost of every primitive with
   [`da4ml`](https://github.com/calad0i/da4ml) (Distributed Arithmetic for ML),
5. **statically schedules** the folded tasks cycle-by-cycle,
6. reports area / latency / throughput / synchronisation metrics, and
7. **proves the folded schedule is numerically equivalent** to the original
   unfolded model (symbolic correctness check).

The graphs and schedules are visualised interactively in the browser via the
[`heterograph`](https://github.com/custom-computing-ic/heterograph) library.

---

## Why this project exists

JEDI-linear and similar jet-tagging networks are compiled to FPGAs as
**fully spatial (unrolled)** dataflow circuits: every multiply gets its own
hardware. That is extremely fast (new event every clock) but very expensive in
LUTs, so the largest / most accurate models do not fit on real devices at low
latency.

**Folding** trades latency and throughput for area: instead of `N` parallel
lanes you build `P < N` lanes and stream the `N` work units through them over
`T = N/P` cycles. The open question this project attacks is:

> *For a given model, what does folding actually cost and buy on real hardware,
> and is the folded circuit still correct?*

Running full Vivado synthesis for every fold factor is far too slow to explore.
STAIR instead builds an **intermediate representation + analytical cost model +
static scheduler** that predicts LUTs, latency, and throughput for any fold
factor in seconds, and verifies correctness symbolically — so you can sweep the
area/latency Pareto front without synthesising anything. `da4ml` is still used to
emit real Verilog when you want ground-truth numbers.

---

## Key terminology

| Term | Meaning |
|------|---------|
| **Logical input** | The full logical tensor consumed by a node (e.g. all `N` particles). |
| **Work unit** | A single element of that tensor, shape `(1 × C)`. |
| **Parallelisation `P` / lanes** | Number of replicated compute lanes across a tensor axis. |
| **Fold factor** | How aggressively a layer is time-multiplexed. `factor=k` → lanes `= N/k`. `factor=1` is fully spatial (unfolded). |
| **Temporal steps `T`** | Cycles needed to *issue* all work for one logical input (`T = N/P`). |
| **Initiation interval `II`** | Cycles before the same kernel can accept a new logical input. |
| **Latency** | `L_pipeline + (T − 1)` — cycles for one logical input to produce an output. |

---

## Repository layout

```
CMIR/
├── main.py                     # End-to-end demo pipeline + web viewer (start here)
├── IR/                         # The STAIR compiler
│   ├── nn_ir/                  # NN-IR: Keras model  ->  hardware graph
│   │   ├── builder.py          #   build_nn_ir(model)
│   │   └── hgq2/               #   HGQ2 quantizer / attention extraction
│   └── sched_ir/               # Sched-IR: the folding + scheduling + cost engine
│       ├── api.py              #   PUBLIC API: evaluate_folded_design, check_symbolic_correctness
│       ├── lowering/           #   NN-IR -> primitive decomposition
│       ├── planning/           #   fold plans
│       ├── scheduling/         #   static scheduler, metrics, sync analysis
│       ├── backends/da4ml/     #   da4ml cost model + Verilog executor
│       └── correctness/        #   symbolic equivalence checker
├── official_models/            # Pretrained .keras weights (git-ignored — see Setup)
├── notebooks/                  # Result generation & Pareto sweeps (deepset, linformer, …)
├── third_party/                # Vendored source of the external deps (see Setup)
│   ├── JEDI-linear/            #   the jet-tagging models + official_models.tar.gz
│   └── imperial_cc_heterograph/#   the graph / web-viewer library
├── docs/superpowers/           # Design docs & implementation plans (local-only, not in git)
└── graphs/                     # PNG exports from main.py --save-graphs (generated)
```

---

## Setup

Everything runs inside a single conda environment. The reference environment on
this machine is called **`jedi-linear`** and already contains the four external
dependencies below. To recreate it from scratch on a new machine:

### 1. Clone and create the environment

```bash
git clone https://github.com/SriV04/CMIR.git
cd CMIR

# Base scientific stack: python 3.13, jax, keras, graph-tool, verilator, …
conda env create -f third_party/JEDI-linear/environment.yml -n jedi-linear
conda activate jedi-linear
```

> Note: `third_party/JEDI-linear/environment.yml` declares `name: m4r` and pins
> `da4ml < 0.5`. Override the name with `-n jedi-linear` as shown, and note the
> working machine actually runs **`da4ml 0.6.x`** (see step 2) — the `< 0.5` pin
> is stale and can be relaxed.

### 2. Install the four core dependencies

These are the pieces that make the toolchain work. Install them **into the
`jedi-linear` env**. On this machine they resolve to the versions in parentheses.

1. **`da4ml`** — hardware cost model + Verilog backend *(0.6.0.dev)*
   ```bash
   pip install "da4ml>=0.6"
   ```

2. **`hgq` / HGQ2** — the quantized-layer library the models are built with
   *(imports as `hgq`, pip package `HGQ2`, 0.1.8)*
   ```bash
   pip install HGQ2
   ```

3. **`heterograph`** — graph data structure + Flask/D3 web viewer.
   Install the vendored copy editable (imports as `heterograph`, pip package
   `imperial-cc-heterograph`). Upstream:
   <https://github.com/custom-computing-ic/heterograph>
   ```bash
   pip install -e third_party/imperial_cc_heterograph
   ```

4. **JEDI-linear models** — the pretrained weights are shipped as a tarball.
   Extract them into `official_models/` at the repo root (this directory is
   git-ignored and is where `main.py` / the notebooks look):
   ```bash
   tar -xvf third_party/JEDI-linear/official_models.tar.gz -C official_models/
   # (create official_models/ first if it does not exist: mkdir -p official_models)
   ```

### 3. Verify the install

```bash
conda activate jedi-linear
python -c "import da4ml, hgq, heterograph, keras, jax; print('ok')"
```

All model loading and the JAX backend require the environment variable
`KERAS_BACKEND=jax` (see below).

---

## Running the demo pipeline

`main.py` runs the whole flow on one model and opens the interactive viewer:

```bash
KERAS_BACKEND=jax python main.py

# Also dump the NN-IR / scheduled / task graphs as PNGs into ./graphs
KERAS_BACKEND=jax python main.py --save-graphs
```

It loads a model (edit the `model_path` near the bottom of `main.py` to switch
between the linformer / deepset / jet-classifier checkpoints), builds the NN-IR,
evaluates a folded design (`fold_factor=2` by default), prints the metrics, runs
the symbolic correctness check, and serves three linked graphs at
**<http://127.0.0.1:8888>**:

- **NN-IR Graph** — the model as a hardware graph.
- **Scheduled Graph** — the folded Sched-IR primitives.
- **Task Graph** — the cycle-accurate static schedule.

The notebooks in `notebooks/` (e.g. `linformer_fold_sweep.ipynb`,
`deepset_results.ipynb`, `stair_evaluation_graphs.ipynb`) drive the same API over
fold-factor sweeps to produce the Pareto plots and the `*_summary.csv` files.

---

## The Python API

Two modules make up the public surface. A minimal end-to-end use:

```python
import keras
from IR import nn_ir
from IR.sched_ir import api

# 1. Load a trained, quantized Keras model
model = keras.models.load_model("official_models/linformers/lin32part.keras")

# 2. Lift it into the NN-IR hardware graph
nn_graph = nn_ir.build_nn_ir(model)

# 3. Fold, cost, and statically schedule it
design = api.evaluate_folded_design(
    nn_graph,
    model=model,
    backend="da4ml",       # only backend currently implemented
    factor=2,              # fold factor (lanes = N / factor); or pass lanes=...
    target_fmax_hz=300e6,  # target clock, used for throughput in Hz
)

# 4. Inspect the results (see "Output object" below)
for k, v in design.metrics.items():
    print(k, v)

# 5. Prove the folded schedule matches the original model numerically
report = api.check_symbolic_correctness(design, model=model)
assert report.passed
```

### `IR.nn_ir.build_nn_ir(model) -> HGraph`
Traverses the Keras model (including HGQ2 quantizers and attention layers) and
returns an **NN-IR** `heterograph.HGraph`: nodes are layers/ops carrying shape,
precision, and quantization metadata; edges carry the tensor transports.

### `IR.sched_ir.api.evaluate_folded_design(nn_graph, *, model, backend="da4ml", factor=None, lanes=None, target_fmax_hz=300e6, resource_config=None) -> EvaluatedDesign`
The core entry point. Internally it: decomposes NN-IR → Sched-IR primitives,
builds a **fold plan** (from `factor` *or* `lanes`), asks the `da4ml` backend for
per-primitive LUT/FF/DSP/BRAM costs, expands primitives into per-cycle **tasks**,
runs the **static scheduler**, computes **metrics**, and runs a
**synchronisation analysis**. Returns an `EvaluatedDesign` (below).

### `IR.sched_ir.api.check_symbolic_correctness(design, *, model, config=None) -> CorrectnessReport`
Builds a symbolic reference trace of the original model and a symbolic trace of
the folded schedule, then checks they produce identical quantized outputs.
Returns a `CorrectnessReport`.

---

## The output object

`evaluate_folded_design` returns an `EvaluatedDesign` dataclass
(`IR/sched_ir/api.py`):

| Field | Type | What it is |
|-------|------|-----------|
| `sched_graph` | `HGraph` | The folded Sched-IR primitive graph (viewable). |
| `fold_plan` | object | The chosen lanes/temporal-steps per layer. |
| `evaluation` | object | Raw da4ml backend result, incl. `.graph` with per-resource cost. |
| `task_ir` | object | Cycle-level task IR (`.resources` carry `.cost` dicts). |
| `task_graph` | `HGraph` | The static schedule as a viewable graph. |
| `task_schedule` | object | `.tasks[id]` → scheduled item with `.start`, `.end`, `.task.ii`, `.task.resource_id`. |
| `sync_report` | object | Synchronisation / buffer-wait analysis. |
| `metrics` | `dict` | Headline numbers (below). |

### `design.metrics`

```python
{
  "area_proxy":                 float,  # total LUTs across all resources
  "work_cost":                  float,  # LUT-cycles of useful work
  "work_to_area_ratio":         float,  # utilisation proxy
  "latency_cycles":             int,    # end-to-end latency of one input
  "sample_ii_cycles":           int,    # initiation interval (throughput bottleneck)
  "throughput_samples_per_sec": float,  # fclk / sample_ii_cycles
  "resource_utilisation":       dict,   # per-resource busy fraction
  "max_resource_utilisation":   float,
  # synchronisation analysis:
  "sync_point_count":           int,
  "max_sync_wait_cycles":       int,
  "total_sync_wait_cycles":     int,
  "mean_sync_wait_cycles":      float,
  "max_buffer_wait_cycles":     int,
  "total_input_buffer_wait_cycles": int,
  "sync_legality_passed":       bool,
  "sync_violations":            int,
}
```

`main.py` also shows how to walk `design.task_schedule` to build a **per-task
cost table** (LUT/FF/DSP/BRAM, start/end cycle, duration, II) as a pandas
DataFrame — this is the data behind the notebook plots.

### `CorrectnessReport`
Returned by `check_symbolic_correctness` (`IR/sched_ir/correctness/records.py`):

| Field | Meaning |
|-------|---------|
| `passed` | `True` iff `failures` is empty (property). |
| `failures` | list of `CorrectnessFailure` (output index, expected vs actual, reason, provenance). |
| `checked_output_count` | number of outputs compared. |
| `reference_qints` / `scheduled_qints` | quantized intervals of reference vs scheduled outputs. |
| `metadata` | free-form provenance. |

---

## Available models

Extracted into `official_models/`, grouped by feature set / architecture:

- `3-feature/`, `3-feature-perminv/` — JEDI-linear GNN, `pt/eta/phi` only
  (perminv = permutation-invariant).
- `16-feature/`, `16-feature-perminv/` — full 16-feature JEDI-linear GNN, for
  `8 … 128` particles.
- `deepset/` — Deep Sets baselines.
- `linformers/` — Linformer attention variants (`lin8part`, `lin16part`,
  `lin32part`).

Each model directory contains the `.keras` checkpoint(s) and, where synthesised,
the `da4ml_verilog_prjs/` with generated Verilog + Vivado reports.

To (re)build a JEDI-linear model architecture in code rather than loading a
checkpoint, use `third_party/JEDI-linear/src/model.py` (`get_gnn`, `get_mlp`,
`get_model`).

---

## Testing

The compiler has an extensive pytest suite (decomposition, folding, scheduling,
cost model, correctness):

```bash
conda activate jedi-linear
KERAS_BACKEND=jax pytest IR/ test_main_graph_export.py   # unit + integration tests
```

There are also notebook-execution smoke tests in `IR/tests/`.

Please email me at sriharsha.vitta@gmail.com if you require assistance, want to contribute to this work, or would like a copy of my findings using STAIR :)
