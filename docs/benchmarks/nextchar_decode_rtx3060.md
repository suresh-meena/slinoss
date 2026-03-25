# NextChar Decode Benchmarks (RTX 3060)

Device:

- `NVIDIA GeForce RTX 3060`
- `sm_86`

Model family:

- `vocab_size=4096`
- `block_size=512`
- `d_model=256`
- `n_layers=6`
- `d_state=64`
- `expand=2`
- `d_head=64`
- `d_conv=4`
- `chunk_size=32`
- `dtype=fp16`

Method:

- `scripts/perf/bench_nextchar_decode.py`
- `warmup_tokens=16`
- `active_tokens=256`
- `repeat=5`
- `backend=cute`

Results:

| B | persistent us/token | eager us/token | speedup | t_lower us/token | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 384.043 | 3190.416 | 8.307x | 4.599 | 0.012 |
| 2 | 258.950 | 1843.955 | 7.121x | 9.198 | 0.036 |
| 4 | 127.536 | 879.899 | 6.899x | 18.394 | 0.144 |
| 8 | 64.503 | 427.166 | 6.622x | 36.786 | 0.570 |
| 16 | 37.810 | 219.635 | 5.809x | 73.570 | 1.946 |

Interpretation:

- The persistent path is materially faster than the eager token loop across the
  entire supported batch grid.
- Relative to the first issue-8 landing, persistent decode improved by about
  `1.12x` at `B=1`, `1.11x` at `B=2`, `1.29x` at `B=4`, `1.52x` at `B=8`, and
  `2.25x` at `B=16`.
- Nsight Compute on the direct `B=16` recurrent kernel dropped from about
  `151.6 us` to `37.4 us` after switching decode to the transposed physical
  state layout and compiling the kernel against the real state strides.
- Nsight Systems on the current `B=1` whole-model persistent path still shows
  the CuTe recurrent decode kernel as the largest GPU bucket at about `34.3%`,
  followed by small projection GEMMs and decode-step conv.
- Small batches are still far from the proxy bound on this GPU.
- The `B=16` efficiency proxy exceeds `1.0`, which means the simple HBM traffic
  model is now overcounting off-chip traffic on this steady-state path. Treat
  the reported efficiency here as a comparative proxy, not a strict physical
  bound, on this card.
- No H100 was available in the local environment, so this table is not an H100
  claim. The decode harness accepts an H100 preset for target-platform runs.
