#!/usr/bin/env python3
"""Benchmark synthetic throughput for SLinOSS, Mamba2, LinOSS, and D-LinOSS."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    ContinuousMamba2Model,
    ContinuousSLinOSSModel,
    count_torch_parameters,
)
from utils import (  # noqa: E402
    CaseSpec,
    filter_cases,
    filter_models,
    load_config,
    make_run_dir,
    run_metadata,
    timing_payload,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="Benchmark configuration YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for experiment.output_root.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output directory name under output_root.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model keys from config.yaml.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of case names from config.yaml.",
    )
    parser.add_argument(
        "--damped-linoss-root",
        type=Path,
        default=None,
        help="Path to a checkout of https://github.com/jaredbmit/damped-linoss.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _resolve_torch_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for torch benchmarks, but CUDA is unavailable.")
    return device


def _configure_torch_backend(*, allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high")


def _sync_torch(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _time_torch_call(fn: Any, *, device: torch.device) -> tuple[float, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        _sync_torch(device)
        stream = torch.cuda.current_stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        out = fn()
        end.record(stream)
        _sync_torch(device)
        return float(start.elapsed_time(end)), out

    started = time.perf_counter()
    out = fn()
    ended = time.perf_counter()
    return (ended - started) * 1000.0, out


def _torch_batches(
    case: CaseSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    total_steps: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(total_steps):
        x = torch.randn(
            (case.batch_size, case.seq_len, case.input_dim),
            generator=generator,
            dtype=dtype,
        ).to(device=device)
        y = torch.randn(
            (case.batch_size, case.seq_len, case.output_dim),
            generator=generator,
            dtype=dtype,
        ).to(device=device)
        batches.append((x, y))
    return batches


def _build_torch_model(
    *,
    case: CaseSpec,
    model_name: str,
    model_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    family = str(model_cfg["family"])
    if family == "slinoss":
        model = ContinuousSLinOSSModel(
            input_dim=case.input_dim,
            hidden_dim=case.hidden_dim,
            output_dim=case.output_dim,
            layers=case.layers,
            d_state=case.state_dim,
            expand=int(model_cfg.get("expand", case.expand)),
            d_head=case.d_head,
            d_conv=int(model_cfg.get("d_conv", case.d_conv)),
            chunk_size=case.chunk_size,
            normalize_bc=bool(model_cfg.get("normalize_bc", True)),
            backend=str(model_cfg.get("backend", "auto")),
        )
        return model.to(device=device, dtype=dtype)

    if family == "mamba2":
        model = ContinuousMamba2Model(
            input_dim=case.input_dim,
            hidden_dim=case.hidden_dim,
            output_dim=case.output_dim,
            layers=case.layers,
            d_state=case.state_dim,
            d_conv=int(model_cfg.get("d_conv", case.d_conv)),
            expand=int(model_cfg.get("expand", case.expand)),
        )
        return model.to(device=device, dtype=dtype)

    raise ValueError(f"Unsupported torch model family for {model_name}: {family}")


def _torch_measure(
    *,
    measure: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    initial_model_state: dict[str, Any],
    initial_optimizer_state: dict[str, Any],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    case: CaseSpec,
) -> dict[str, Any]:
    _require(len(batches) >= 1 + warmup_steps + measure_steps, "Insufficient batches.")

    def run_once(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        if measure == "forward":
            model.eval()
            with torch.no_grad():
                return model(xb)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(xb)
        loss = F.mse_loss(output, yb)
        loss.backward()
        if measure == "train_step":
            optimizer.step()
        return loss.detach()

    model.load_state_dict(copy.deepcopy(initial_model_state))
    optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
    cold_ms, _ = _time_torch_call(
        lambda: run_once(*batches[0]),
        device=device,
    )

    model.load_state_dict(copy.deepcopy(initial_model_state))
    optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
    for xb, yb in batches[1 : 1 + warmup_steps]:
        run_once(xb, yb)
    _sync_torch(device)

    warm_samples: list[float] = []
    for xb, yb in batches[1 + warmup_steps : 1 + warmup_steps + measure_steps]:
        sample_ms, _ = _time_torch_call(
            lambda xb=xb, yb=yb: run_once(xb, yb),
            device=device,
        )
        warm_samples.append(sample_ms)

    return timing_payload(
        warm_samples,
        sequences_per_step=case.sequences_per_step,
        timesteps_per_step=case.timesteps_per_step,
        cold_ms=cold_ms,
    )


def benchmark_torch_model(
    *,
    model_name: str,
    model_cfg: dict[str, Any],
    case: CaseSpec,
    experiment_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    torch_device = _resolve_torch_device(str(experiment_cfg["torch_device"]))
    torch_dtype_name = str(experiment_cfg["torch_dtype"])
    torch_dtype = _torch_dtype_from_name(torch_dtype_name)
    warmup_steps = int(experiment_cfg["warmup_steps"])
    measure_steps = int(experiment_cfg["measure_steps"])
    total_steps = 1 + warmup_steps + measure_steps
    optimizer_cfg = dict(experiment_cfg["optimizer"])

    model = _build_torch_model(
        case=case,
        model_name=model_name,
        model_cfg=model_cfg,
        device=torch_device,
        dtype=torch_dtype,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    param_count, param_bytes = count_torch_parameters(model)
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    batches = _torch_batches(
        case,
        device=torch_device,
        dtype=torch_dtype,
        total_steps=total_steps,
        seed=seed,
    )

    measurements = {
        measure: _torch_measure(
            measure=measure,
            model=model,
            optimizer=optimizer,
            initial_model_state=initial_model_state,
            initial_optimizer_state=initial_optimizer_state,
            batches=batches,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=torch_device,
            case=case,
        )
        for measure in ("forward", "backward", "train_step")
    }
    return {
        "framework": "pytorch",
        "family": str(model_cfg["family"]),
        "device": str(torch_device),
        "dtype": torch_dtype_name,
        "params": {
            "count": int(param_count),
            "num_bytes": int(param_bytes),
            "millions": float(param_count / 1e6),
            "megabytes": float(param_bytes / (1024.0**2)),
        },
        "model_config": copy.deepcopy(model_cfg),
        "measurements": measurements,
    }


class DampedLinossApi:
    def __init__(self, root: Path) -> None:
        src_root = root / "src"
        if not src_root.is_dir():
            raise FileNotFoundError(
                f"Expected damped-linoss source tree under {src_root}."
            )
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        create_model_mod = importlib.import_module("damped_linoss.models.create_model")
        train_mod = importlib.import_module("damped_linoss.train")
        self.jax = importlib.import_module("jax")
        self.jnp = importlib.import_module("jax.numpy")
        self.jr = importlib.import_module("jax.random")
        self.create_model = create_model_mod.create_model
        self.create_optimizer = train_mod.create_optimizer
        self.calc_output = train_mod.calc_output
        self.regression_loss = train_mod.regression_loss
        self.make_step = train_mod.make_step
        self.count_params = train_mod.count_params


def _resolve_damped_linoss_root(arg_root: Path | None) -> Path:
    if arg_root is not None:
        return arg_root.resolve()
    env_root = os.environ.get("DAMPED_LINOSS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    raise RuntimeError(
        "Damped-LinOSS checkout is required. Pass --damped-linoss-root or set DAMPED_LINOSS_ROOT."
    )


def _resolve_jax_dtype(api: DampedLinossApi, name: str) -> Any:
    if name == "float16":
        return api.jnp.float16
    if name == "bfloat16":
        return api.jnp.bfloat16
    if name == "float32":
        return api.jnp.float32
    raise ValueError(f"Unsupported JAX dtype: {name}")


def _resolve_jax_device(api: DampedLinossApi, platform: str) -> Any:
    devices = api.jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX devices available for platform={platform!r}.")
    return devices[0]


def _block_jax_tree(api: DampedLinossApi, value: Any) -> Any:
    def block_leaf(leaf: Any) -> Any:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
        return leaf

    api.jax.tree_util.tree_map(block_leaf, value)
    return value


def _time_jax_call(api: DampedLinossApi, fn: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    out = fn()
    _block_jax_tree(api, out)
    ended = time.perf_counter()
    return (ended - started) * 1000.0, out


def _jax_batches(
    api: DampedLinossApi,
    case: CaseSpec,
    *,
    dtype: Any,
    device: Any,
    total_steps: int,
    seed: int,
) -> list[tuple[Any, Any]]:
    key = api.jr.PRNGKey(seed)
    batches: list[tuple[Any, Any]] = []
    for _ in range(total_steps):
        key, x_key, y_key = api.jr.split(key, 3)
        x = api.jr.normal(
            x_key,
            (case.batch_size, case.seq_len, case.input_dim),
            dtype=dtype,
        )
        y = api.jr.normal(
            y_key,
            (case.batch_size, case.seq_len, case.output_dim),
            dtype=dtype,
        )
        batches.append((api.jax.device_put(x, device), api.jax.device_put(y, device)))
    return batches


def _build_jax_hparams(case: CaseSpec, model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": "LinOSS",
        "layer_name": str(model_cfg["layer_name"]),
        "input_dim": int(case.input_dim),
        "state_dim": int(case.state_dim),
        "hidden_dim": int(case.hidden_dim),
        "output_dim": int(case.output_dim),
        "num_blocks": int(case.layers),
        "classification": False,
        "tanh_output": False,
        "output_step": 1,
        "initialization": str(model_cfg.get("initialization", "ring")),
        "r_min": float(model_cfg.get("r_min", 0.9)),
        "r_max": float(model_cfg.get("r_max", 1.0)),
        "theta_min": float(model_cfg.get("theta_min", 0.0)),
        "theta_max": float(model_cfg.get("theta_max", 3.141592653589793)),
        "A_min": float(model_cfg.get("A_min", 0.0)),
        "A_max": float(model_cfg.get("A_max", 1.0)),
        "G_min": float(model_cfg.get("G_min", 0.0)),
        "G_max": float(model_cfg.get("G_max", 1.0)),
        "dt_std": float(model_cfg.get("dt_std", 0.5)),
        "drop_rate": float(model_cfg.get("drop_rate", 0.0)),
    }


def _jax_step(
    api: DampedLinossApi,
    *,
    measure: str,
    model: Any,
    state: Any,
    opt: Any,
    opt_state: Any,
    xb: Any,
    yb: Any,
    key: Any,
) -> tuple[Any, Any, Any, Any]:
    if measure == "forward":
        output, next_state = api.calc_output(
            model,
            xb,
            state,
            key,
            model.stateful,
            model.nondeterministic,
        )
        return model, next_state, opt_state, output

    if measure == "backward":
        (value, next_state), grads = api.regression_loss(model, xb, yb, state, key)
        return model, next_state, opt_state, (value, grads)

    if measure == "train_step":
        next_model, next_state, next_opt_state, value = api.make_step(
            model,
            xb,
            yb,
            api.regression_loss,
            state,
            opt,
            opt_state,
            key,
        )
        return next_model, next_state, next_opt_state, value

    raise ValueError(f"Unsupported JAX measure: {measure}")


def _jax_measure(
    api: DampedLinossApi,
    *,
    measure: str,
    model: Any,
    state: Any,
    opt: Any,
    opt_state: Any,
    batches: list[tuple[Any, Any]],
    warmup_steps: int,
    measure_steps: int,
    case: CaseSpec,
    seed: int,
) -> dict[str, Any]:
    _require(len(batches) >= 1 + warmup_steps + measure_steps, "Insufficient batches.")

    key = api.jr.PRNGKey(seed)
    cold_key, key = api.jr.split(key)
    cold_ms, _ = _time_jax_call(
        api,
        lambda: _jax_step(
            api,
            measure=measure,
            model=model,
            state=state,
            opt=opt,
            opt_state=opt_state,
            xb=batches[0][0],
            yb=batches[0][1],
            key=cold_key,
        ),
    )

    run_model = model
    run_state = state
    run_opt_state = opt_state
    for xb, yb in batches[1 : 1 + warmup_steps]:
        step_key, key = api.jr.split(key)
        out = _jax_step(
            api,
            measure=measure,
            model=run_model,
            state=run_state,
            opt=opt,
            opt_state=run_opt_state,
            xb=xb,
            yb=yb,
            key=step_key,
        )
        _block_jax_tree(api, out)
        run_model, run_state, run_opt_state, _ = out

    warm_samples: list[float] = []
    for xb, yb in batches[1 + warmup_steps : 1 + warmup_steps + measure_steps]:
        step_key, key = api.jr.split(key)

        def step_fn() -> tuple[Any, Any, Any, Any]:
            return _jax_step(
                api,
                measure=measure,
                model=run_model,
                state=run_state,
                opt=opt,
                opt_state=run_opt_state,
                xb=xb,
                yb=yb,
                key=step_key,
            )

        sample_ms, out = _time_jax_call(api, step_fn)
        run_model, run_state, run_opt_state, _ = out
        warm_samples.append(sample_ms)

    return timing_payload(
        warm_samples,
        sequences_per_step=case.sequences_per_step,
        timesteps_per_step=case.timesteps_per_step,
        cold_ms=cold_ms,
    )


def benchmark_jax_model(
    api: DampedLinossApi,
    *,
    model_cfg: dict[str, Any],
    case: CaseSpec,
    experiment_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    jax_device = _resolve_jax_device(api, str(experiment_cfg["jax_platform"]))
    jax_dtype_name = str(experiment_cfg["jax_dtype"])
    jax_dtype = _resolve_jax_dtype(api, jax_dtype_name)
    warmup_steps = int(experiment_cfg["warmup_steps"])
    measure_steps = int(experiment_cfg["measure_steps"])
    total_steps = 1 + warmup_steps + measure_steps
    optimizer_cfg = dict(experiment_cfg["optimizer"])

    model_key = api.jr.PRNGKey(seed)
    model, state = api.create_model(_build_jax_hparams(case, model_cfg), model_key)
    model = api.jax.device_put(model, jax_device)
    state = api.jax.device_put(state, jax_device)
    opt, opt_state = api.create_optimizer(
        model,
        num_steps=total_steps,
        lr=float(optimizer_cfg["lr"]),
        ssm_lr_factor=1.0,
        weight_decay=float(optimizer_cfg["weight_decay"]),
        use_warmup_cosine=False,
    )
    opt_state = api.jax.device_put(opt_state, jax_device)
    param_count, param_bytes = api.count_params(model)
    batches = _jax_batches(
        api,
        case,
        dtype=jax_dtype,
        device=jax_device,
        total_steps=total_steps,
        seed=seed,
    )

    measurements = {
        measure: _jax_measure(
            api,
            measure=measure,
            model=model,
            state=state,
            opt=opt,
            opt_state=opt_state,
            batches=batches,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            case=case,
            seed=seed,
        )
        for measure in ("forward", "backward", "train_step")
    }
    return {
        "framework": "jax",
        "family": str(model_cfg["family"]),
        "device": str(jax_device),
        "dtype": jax_dtype_name,
        "params": {
            "count": int(param_count),
            "num_bytes": int(param_bytes),
            "millions": float(param_count / 1e6),
            "megabytes": float(param_bytes / (1024.0**2)),
        },
        "model_config": copy.deepcopy(model_cfg),
        "measurements": measurements,
    }


def _mamba_version() -> str | None:
    try:
        module = importlib.import_module("mamba_ssm")
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    experiment_cfg = dict(config["experiment"])
    selected_models = filter_models(
        config["models"],
        None if args.models is None else set(args.models),
    )
    selected_cases = filter_cases(
        config["cases"],
        None if args.cases is None else set(args.cases),
    )
    _require(selected_models, "No models selected.")
    _require(selected_cases, "No benchmark cases selected.")

    _set_seed(int(experiment_cfg["seed"]))
    _configure_torch_backend(allow_tf32=bool(experiment_cfg.get("allow_tf32", True)))

    needs_damped_linoss = any(
        str(model_cfg["family"]) == "damped_linoss"
        for model_cfg in selected_models.values()
    )
    damped_api = None
    damped_root = None
    if needs_damped_linoss:
        damped_root = _resolve_damped_linoss_root(args.damped_linoss_root)
        damped_api = DampedLinossApi(damped_root)

    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(str(experiment_cfg["output_root"]))
    )
    run_dir = make_run_dir(output_root, run_name=args.run_name)
    results_path = run_dir / "results.json"

    results: dict[str, Any] = {
        "kind": "throughput_comparison",
        "schema_version": 1,
        "metadata": {
            **run_metadata(),
            "config_path": str(args.config.resolve()),
            "damped_linoss_root": None if damped_root is None else str(damped_root),
            "framework_versions": {
                "torch": torch.__version__,
                "mamba_ssm": _mamba_version(),
                "jax": None if damped_api is None else damped_api.jax.__version__,
            },
        },
        "experiment": copy.deepcopy(experiment_cfg),
        "selected_models": copy.deepcopy(selected_models),
        "cases": {},
    }

    for case_index, case in enumerate(selected_cases):
        case_seed = int(experiment_cfg["seed"]) + case_index * 101
        case_payload: dict[str, Any] = {
            "case": case.payload,
            "models": {},
        }
        for model_index, (model_name, model_cfg) in enumerate(selected_models.items()):
            seed = case_seed + model_index * 1009
            family = str(model_cfg["family"])
            print(f"[bench] case={case.name} model={model_name}", flush=True)
            if family in {"slinoss", "mamba2"}:
                model_result = benchmark_torch_model(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    case=case,
                    experiment_cfg=experiment_cfg,
                    seed=seed,
                )
            elif family == "damped_linoss":
                if damped_api is None:
                    raise RuntimeError("Damped-LinOSS API was not initialized.")
                model_result = benchmark_jax_model(
                    damped_api,
                    model_cfg=model_cfg,
                    case=case,
                    experiment_cfg=experiment_cfg,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported model family: {family}")
            case_payload["models"][model_name] = model_result
        results["cases"][case.name] = case_payload

    write_json(results_path, results)
    print(f"[done] results={results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
