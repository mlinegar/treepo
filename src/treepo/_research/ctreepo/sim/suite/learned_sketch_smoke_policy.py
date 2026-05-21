from __future__ import annotations

from dataclasses import asdict, dataclass

from treepo._research.ctreepo.sim.suite.policy_common import parse_float_list, parse_int_list


@dataclass(frozen=True)
class LearnedSketchSmokePolicy:
    state_dims: tuple[int, ...] = (16,)
    train_sizes: tuple[int, ...] = (32,)
    zipf_alphas: tuple[float, ...] = (1.0,)
    n_val: int = 16
    n_test: int = 32
    hidden_dim: int = 32
    n_epochs: int = 2
    batch_size: int = 8
    device: str = "cpu"
    torch_threads: int = 1
    seed: int = 0
    simulation_mode: str = "latent_proxy_baseline"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_learned_sketch_smoke_policy(
    *,
    state_dims: str | None = None,
    train_sizes: str | None = None,
    zipf_alphas: str | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    hidden_dim: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    torch_threads: int | None = None,
    seed: int | None = None,
    simulation_mode: str | None = None,
) -> LearnedSketchSmokePolicy:
    defaults = LearnedSketchSmokePolicy()
    return LearnedSketchSmokePolicy(
        state_dims=parse_int_list(state_dims, default=defaults.state_dims),
        train_sizes=parse_int_list(train_sizes, default=defaults.train_sizes),
        zipf_alphas=parse_float_list(zipf_alphas, default=defaults.zipf_alphas),
        n_val=int(defaults.n_val if n_val is None else n_val),
        n_test=int(defaults.n_test if n_test is None else n_test),
        hidden_dim=int(defaults.hidden_dim if hidden_dim is None else hidden_dim),
        n_epochs=int(defaults.n_epochs if n_epochs is None else n_epochs),
        batch_size=int(defaults.batch_size if batch_size is None else batch_size),
        device=str(defaults.device if device is None else device),
        torch_threads=int(defaults.torch_threads if torch_threads is None else torch_threads),
        seed=int(defaults.seed if seed is None else seed),
        simulation_mode=str(defaults.simulation_mode if simulation_mode is None else simulation_mode),
    )


__all__ = [
    "LearnedSketchSmokePolicy",
    "resolve_learned_sketch_smoke_policy",
]
