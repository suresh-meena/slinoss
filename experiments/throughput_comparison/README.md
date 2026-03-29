# Throughput Comparison Experiment

This experiment benchmarks sequence-model throughput across:

- `SLinOSS` from this repo
- `Mamba2` from `mamba-ssm`
- `LinOSS` from the upstream JAX repo
- `D-LinOSS` from the upstream JAX repo

It uses a synthetic-token nextchar contract: models consume integer token
sequences and optimize next-token cross-entropy. This keeps the workload close to
`examples/nextchar.py` while remaining dataset-independent and repeatable.

## Layout

- `config.yaml`: Benchmark defaults, model registry, and paper-oriented case grid.
- `run.py`: Cross-framework throughput runner that writes JSON.
- `models.py`: PyTorch wrappers for `SLinOSS` and `Mamba2`.
- `utils.py`: Config loading, validation, summaries, and JSON helpers.
- `analysis.py`: Generates CSV, markdown tables, and plots from a benchmark run.
- `requirements.txt`: Extra dependencies kept local to this experiment only.

## Methodology

All models are benchmarked on the same nextchar training contract:

- input: `(batch, seq_len)` integer token ids
- target: `(batch, seq_len)` shifted next-token ids
- logits: `(batch, seq_len, vocab_size)`
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

The JAX LinOSS / D-LinOSS path is implemented locally in this experiment under
`experiments/throughput_comparison/dlinoss_jax.py`.

## Usage

Run the benchmark:

```bash
python experiments/throughput_comparison/run.py \
  --config experiments/throughput_comparison/config.yaml
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
- For this experiment, `input_dim` and `output_dim` act as the shared vocab size
  and must match in every case.
- Small JAX smoke cases may emit `cuda_timer.cc:87` "Delay kernel timed out"
  warnings. These are timing-accuracy warnings from JAX/XLA on tiny kernels, not
  correctness failures. For cleaner timing logs, prefer larger benchmark cases
  (for example `config.yaml`) or profile with Nsight Systems.
