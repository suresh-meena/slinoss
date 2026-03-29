# Language Modeling

This experiment adds a FineWeb-Edu language-modeling path for `SLinOSS` using
Hugging Face `accelerate`, resumable checkpoints, and structured run logging.

The training recipe is aligned with the papers the repo points to:

- `Mamba-3` data/training setup: FineWeb-Edu, 100B-token budget, `Llama-3.1`
  tokenizer, 2K context length, and bf16 pretraining.
- `Mamba-2` optimizer recipe: AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.1`,
  `grad_clip=1.0`, no dropout, and linear warmup followed by cosine decay.

Two things are intentionally different from paper-exact `Mamba-3`:

1. The `SLinOSS` model keeps the proposal-defined project/conv/scan mixer from
   [`main-5.pdf`](../../main-5.pdf) instead of replacing the conv path with the
   `Mamba-3` state-update parameterization.
2. The default `slinoss_180m` preset is parameter-budget matched for this repo's
   current `SLinOSS` implementation, so its width/state sizes are not identical
   to the `Mamba-3` recurrent stack.

## Files

- `run.py`: main Accelerate training entrypoint.
- `model.py`: `SLinOSS` and optional `Mamba2` LM blocks with the same outer LM shell.
- `data.py`: streaming FineWeb-Edu loader plus online token packing.
- `compare_runs.py`: compare multiple `summary.json` outputs.
- `config.yaml`: default config and presets.

## Install

From the repo root:

```bash
pip install -r experiments/language_modeling/requirements.txt
```

If you want the paper tokenizer exactly, make sure your Hugging Face account can
access `meta-llama/Llama-3.1-8B` and export `HF_TOKEN` before launch. If not,
override `data.tokenizer_name=...` with a public tokenizer.

## Launch

Configure Accelerate once:

```bash
accelerate config
```

Run the default 180M-style `SLinOSS` preset:

```bash
accelerate launch experiments/language_modeling/run.py \
  --config experiments/language_modeling/config.yaml \
  --preset slinoss_180m
```

Run the optional `Mamba2` comparison model on the same pipeline:

```bash
accelerate launch experiments/language_modeling/run.py \
  --config experiments/language_modeling/config.yaml \
  --preset mamba2_180m
```

Common overrides:

```bash
accelerate launch experiments/language_modeling/run.py \
  --preset slinoss_180m \
  --set training.max_steps=1000 \
  --set training.gradient_accumulation_steps=16 \
  --set data.tokenizer_name=HuggingFaceTB/SmolLM2-135M-Instruct
```

Resume from a checkpoint:

```bash
accelerate launch experiments/language_modeling/run.py \
  --preset slinoss_180m \
  --resume-from experiments/language_modeling/runs/<run>/checkpoints/step_0010000
```

## Outputs

Each run writes:

- `train.log`: human-readable logs.
- `metrics.jsonl`: step/eval metrics.
- `summary.json`: latest resolved summary for comparison scripts.
- `checkpoints/step_*/`: Accelerate checkpoints with model/optimizer/scheduler/state.

Compare runs:

```bash
python3 experiments/language_modeling/compare_runs.py \
  experiments/language_modeling/runs/run_a \
  experiments/language_modeling/runs/run_b
```

## Default 180M Preset

The default `SLinOSS` preset targets the same high-level pretraining contract as
`Mamba-3` while staying close to a 180M parameter budget for the current repo:

- dataset: `HuggingFaceFW/fineweb-edu`, config `sample-100BT`
- tokenizer: `meta-llama/Llama-3.1-8B`
- context length: `2048`
- token budget: `100B`
- precision: `bf16`
- optimizer: AdamW with `lr=5e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`
- scheduler: `2000` warmup steps then cosine decay to `10%`
- block: `d_model=640`, `n_layers=12`, `d_state=32`, `expand=2`, `d_head=64`,
  `d_conv=4`, `chunk_size=128`, `ffn_hidden_dim=1500`

`run.py` logs the exact trainable parameter count at startup so you can adjust
the preset once the local environment has the full runtime stack installed.
