"""Local JAX LinOSS / D-LinOSS implementation for throughput benchmarks.

This module provides the minimal API used by `run.py` so the benchmark does not
require an external checkout of the upstream damped-linoss repository.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


def _safe_load(data: dict[str, Any], key: str, dtype: type | None = None) -> Any:
    value = data.get(key, None)
    if value is None:
        raise KeyError(f"Key {key!r} does not exist.")
    return value if dtype is None else dtype(value)


class _LinOSSBlock(eqx.Module):
    in_to_state: jax.Array
    state_to_hidden: jax.Array
    direct: jax.Array
    a_diag: jax.Array
    g_diag: jax.Array
    dt: jax.Array
    norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear
    damped: bool = eqx.field(static=True)

    def __call__(self, x: jax.Array, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_norm = self.norm(x)

        dt = jax.nn.softplus(self.dt) + 1e-4
        omega = self.a_diag * dt
        decay = jnp.exp(-jax.nn.softplus(self.g_diag) * dt)
        if self.damped:
            decay = decay * decay

        coeff = decay * jnp.cos(omega)
        next_state = coeff * state + self.in_to_state @ x_norm

        mixed = self.state_to_hidden @ next_state + self.direct * x_norm
        mixed = jax.nn.gelu(mixed)
        return x + self.out_proj(mixed), next_state


class LocalLinOSS(eqx.Module):
    token_embed: jax.Array
    pos_embed: jax.Array
    blocks: tuple[_LinOSSBlock, ...]
    output_proj: eqx.nn.Linear
    vocab_size: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    output_step: int = eqx.field(static=True)
    classification: bool = eqx.field(static=True)
    tanh_output: bool = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    nondeterministic: bool = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        tokens = jnp.asarray(x, dtype=jnp.int32)
        hidden = self.token_embed[tokens]
        hidden = hidden + self.pos_embed[: hidden.shape[0]]
        init_state = jnp.zeros((len(self.blocks), self.state_dim), dtype=hidden.dtype)

        def step_fn(
            carry: jax.Array,
            x_t: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            states = carry
            h = x_t
            next_states: list[jax.Array] = []
            for i, block in enumerate(self.blocks):
                h, state_i = block(h, states[i])
                next_states.append(state_i)
            y_t = self.output_proj(h)
            if self.tanh_output:
                y_t = jnp.tanh(y_t)
            return jnp.stack(next_states), y_t

        _, y = jax.lax.scan(step_fn, init_state, hidden)
        if self.output_step > 1:
            y = y[:: self.output_step]
        return y


def create_model(hyperparameters: dict[str, Any], key: jax.Array) -> tuple[LocalLinOSS, None]:
    model_name = _safe_load(hyperparameters, "model_name", str)
    if model_name != "LinOSS":
        raise ValueError(f"Unsupported model_name: {model_name}")

    layer_name = _safe_load(hyperparameters, "layer_name", str)
    input_dim = _safe_load(hyperparameters, "input_dim", int)
    state_dim = _safe_load(hyperparameters, "state_dim", int)
    hidden_dim = _safe_load(hyperparameters, "hidden_dim", int)
    output_dim = _safe_load(hyperparameters, "output_dim", int)
    vocab_size = int(hyperparameters.get("vocab_size", input_dim))
    seq_len = int(hyperparameters.get("seq_len", 1))
    num_blocks = _safe_load(hyperparameters, "num_blocks", int)

    classification = _safe_load(hyperparameters, "classification", bool)
    tanh_output = _safe_load(hyperparameters, "tanh_output", bool)
    output_step = _safe_load(hyperparameters, "output_step", int)

    r_min = float(hyperparameters.get("r_min", 0.9))
    r_max = float(hyperparameters.get("r_max", 1.0))
    theta_min = float(hyperparameters.get("theta_min", 0.0))
    theta_max = float(hyperparameters.get("theta_max", jnp.pi))
    dt_std = float(hyperparameters.get("dt_std", 0.5))

    damped = layer_name.lower().startswith("damped")

    if input_dim != output_dim:
        raise ValueError(
            f"nextchar requires input_dim == output_dim (vocab), got {input_dim} and {output_dim}."
        )
    if vocab_size != input_dim:
        raise ValueError(
            f"vocab_size must match input_dim for nextchar, got {vocab_size} and {input_dim}."
        )

    keys = jr.split(key, num_blocks * 4 + 3)
    token_embed = 0.02 * jr.normal(keys[0], (vocab_size, hidden_dim))
    pos_embed = 0.01 * jr.normal(keys[1], (seq_len, hidden_dim))
    output_proj = eqx.nn.Linear(hidden_dim, output_dim, key=keys[2])

    blocks: list[_LinOSSBlock] = []
    for i in range(num_blocks):
        k0, k1, k2, k3 = keys[3 + i * 4 : 3 + (i + 1) * 4]
        in_to_state = jr.normal(k0, (state_dim, hidden_dim)) / jnp.sqrt(float(hidden_dim))
        state_to_hidden = jr.normal(k1, (hidden_dim, state_dim)) / jnp.sqrt(float(state_dim))
        direct = jr.uniform(k2, (hidden_dim,), minval=0.0, maxval=1.0)

        a_diag = jr.uniform(k3, (state_dim,), minval=theta_min, maxval=theta_max)
        g_diag = jr.uniform(k2, (state_dim,), minval=-jnp.log(r_max + 1e-6), maxval=-jnp.log(r_min + 1e-6))
        dt = dt_std * jr.normal(k1, (state_dim,))

        blocks.append(
            _LinOSSBlock(
                in_to_state=in_to_state,
                state_to_hidden=state_to_hidden,
                direct=direct,
                a_diag=a_diag,
                g_diag=g_diag,
                dt=dt,
                norm=eqx.nn.LayerNorm(hidden_dim),
                out_proj=eqx.nn.Linear(hidden_dim, hidden_dim, key=k0),
                damped=damped,
            )
        )

    model = LocalLinOSS(
        token_embed=token_embed,
        pos_embed=pos_embed,
        blocks=tuple(blocks),
        output_proj=output_proj,
        vocab_size=vocab_size,
        state_dim=state_dim,
        output_step=output_step,
        classification=classification,
        tanh_output=tanh_output,
        stateful=False,
        nondeterministic=False,
    )
    return model, None


def count_params(model: Any) -> tuple[int, int]:
    leaves, _ = jax.tree_util.tree_flatten(model)
    arrays = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    num_params = int(sum(arr.size for arr in arrays))
    num_bytes = int(sum(arr.size * arr.dtype.itemsize for arr in arrays))
    return num_params, num_bytes


def create_optimizer(
    model: Any,
    num_steps: int,
    lr: float,
    ssm_lr_factor: float,
    weight_decay: float,
    use_warmup_cosine: bool,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    del num_steps, ssm_lr_factor, use_warmup_cosine
    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    return opt, opt_state


@eqx.filter_jit
def calc_output(
    model: Any,
    x: jax.Array,
    state: Any,
    key: jax.Array,
    stateful: bool,
    nondeterministic: bool,
) -> tuple[jax.Array, Any]:
    del key, stateful, nondeterministic
    return jax.vmap(model)(x), state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(
    model: Any,
    x: jax.Array,
    y: jax.Array,
    state: Any,
    key: jax.Array,
) -> tuple[jax.Array, Any]:
    pred_y, state = calc_output(model, x, state, key, model.stateful, model.nondeterministic)
    targets = jnp.asarray(y, dtype=jnp.int32)
    log_probs = jax.nn.log_softmax(pred_y, axis=-1)
    token_logp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    nll = -jnp.mean(jnp.squeeze(token_logp, axis=-1))
    return nll, state


@eqx.filter_jit
def make_step(
    model: Any,
    x: jax.Array,
    y: jax.Array,
    loss_fn: Any,
    state: Any,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jax.Array,
) -> tuple[Any, Any, optax.OptState, jax.Array]:
    (value, state), grads = loss_fn(model, x, y, state, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value
