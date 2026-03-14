# Throughput Comparison Experiment

This experiment benchmarks sequence-model throughput across:

- `SLinOSS` from this repo
- `Mamba2` from `mamba-ssm`
- `LinOSS` from the upstream JAX repo
- `D-LinOSS` from the upstream JAX repo

It is intentionally synthetic. The benchmark uses random continuous sequences and
random regression targets so the comparison stays focused on throughput instead of
dataset effects.

## Layout

- `config.yaml`: Benchmark defaults, model registry, and paper-oriented case grid.
- `run.py`: Cross-framework throughput runner that writes JSON.
- `models.py`: PyTorch wrappers for `SLinOSS` and `Mamba2`.
- `utils.py`: Config loading, validation, summaries, and JSON helpers.
- `analysis.py`: Generates CSV, markdown tables, and plots from a benchmark run.
- `requirements.txt`: Extra dependencies kept local to this experiment only.

## Methodology

All models are benchmarked on the same sequence-to-sequence regression contract:

- input: `(batch, seq_len, input_dim)` floating-point sequence
- target: `(batch, seq_len, output_dim)` floating-point sequence
- measurements:
  - `forward`: inference-only latency/throughput
  - `backward`: forward + loss + gradient computation
  - `train_step`: forward + backward + optimizer update
- timing:
  - `cold_ms`: first measured step, which captures compile/cache effects
  - `warm_*`: steady-state timings after warmup

This setup keeps the benchmark close to a NeurIPS-style systems section:

- main steady-state throughput table
- cold-vs-warm latency table
- sequence-length scaling
- batch-size scaling
- model-width/depth scaling
- throughput-vs-parameter-count scatter

## Dependencies

Install the isolated experiment dependencies from the repo root:

```bash
python3 -m venv .venv-throughput
source .venv-throughput/bin/activate
pip install -r experiments/throughput_comparison/requirements.txt
```

The JAX LinOSS code is imported from a separate checkout of
`https://github.com/jaredbmit/damped-linoss`. Point the runner at that checkout
with `--damped-linoss-root` or `DAMPED_LINOSS_ROOT`.

## Usage

Run the benchmark:

```bash
python experiments/throughput_comparison/run.py \
  --config experiments/throughput_comparison/config.yaml \
  --damped-linoss-root /path/to/damped-linoss
```

Generate the paper-style summary:

```bash
python experiments/throughput_comparison/analysis.py \
  experiments/throughput_comparison/runs/<run_name>/results.json
```

## Notes

- The benchmark keeps root repo dependencies unchanged.
- `SLinOSS` follows the residual-block style used in `examples/nextchar.py`.
- `Mamba2` uses the official `mamba-ssm` layer in a pre-norm residual stack.
- `LinOSS` and `D-LinOSS` use the upstream Equinox implementation directly.
