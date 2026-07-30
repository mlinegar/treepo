from __future__ import annotations

from pathlib import Path

import pytest

from treepo.methods.families import resolve_family


def _config(operator_kind: str) -> dict[str, object]:
    return {
        "operator_kind": operator_kind,
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 9,
    }


@pytest.mark.parametrize("operator_kind", ["fno", "tfno", "uno", "conv1d"])
def test_every_builtin_operator_kind_registers_one_shared_g(operator_kind: str) -> None:
    pytest.importorskip("torch")
    family_name = "fno" if operator_kind == "fno" else "neural_operator"
    family = resolve_family(family_name, _config(operator_kind))
    family._ensure_model(output_dim=1)
    model = family._model
    assert model is not None

    assert model.g_module_names == ("g",)
    assert set(dict(model.named_children())) == {"g", "readout"}
    assert not any(hasattr(model, name) for name in ("leaf_fno", "merge_fno", "leaf_norm"))
    assert not hasattr(model, "g_leaf_module_names")
    assert not hasattr(model, "g_merge_module_names")
    assert all(
        key == "_metadata" or key.startswith(("g.", "readout.")) for key in model.state_dict()
    )


def test_leaf_and_merge_aliases_invoke_the_same_g_module() -> None:
    torch = pytest.importorskip("torch")
    family = resolve_family("neural_operator", _config("conv1d"))
    family._ensure_model(output_dim=1)
    model = family._model
    assert model is not None

    calls: list[int] = []
    handle = model.g.register_forward_hook(lambda module, args, output: calls.append(id(module)))
    try:
        leaves = torch.randn(1, 2, 8)
        leaf_states = model.leaf_operator(leaves)
        merged = model.merge(torch.cat([leaf_states[:, 0], leaf_states[:, 1]], dim=-1))
    finally:
        handle.remove()

    assert calls == [id(model.g), id(model.g)]
    torch.testing.assert_close(
        merged,
        model.g(leaf_states[:, 0], leaf_states[:, 1]),
    )


def test_occupancy_distinguishes_leaf_null_from_zero_right_merge_state() -> None:
    torch = pytest.importorskip("torch")
    family = resolve_family("neural_operator", _config("conv1d"))
    family._ensure_model(output_dim=1)
    g = family._model.g

    left = torch.randn(2, 8)
    zero = torch.zeros_like(left)
    leaf_input = g.pack_inputs(left)
    merge_input = g.pack_inputs(left, zero)

    assert torch.equal(leaf_input[:, :3], merge_input[:, :3])
    assert torch.equal(leaf_input[:, 3], torch.zeros_like(left))
    assert torch.equal(merge_input[:, 3], torch.ones_like(left))
    assert not torch.equal(leaf_input, merge_input)


def test_masked_mean_residual_has_coherent_shared_identity_initialization() -> None:
    torch = pytest.importorskip("torch")
    family = resolve_family("neural_operator", _config("conv1d"))
    family._ensure_model(output_dim=1)
    g = family._model.g

    left = torch.randn(3, 8)
    right = torch.randn(3, 8)
    assert torch.equal(g(left), left)
    assert torch.equal(g(left, right), 0.5 * (left + right))


def test_singleton_and_recursive_tree_reuse_one_g_for_every_logical_node() -> None:
    torch = pytest.importorskip("torch")
    family = resolve_family("neural_operator", _config("conv1d"))
    family._ensure_model(output_dim=1)
    model = family._model
    assert model is not None

    logical_rows: list[tuple[int, bool]] = []

    def record(_module, args, _output) -> None:
        logical_rows.append((int(args[0].reshape(-1, 8).shape[0]), len(args) == 2))

    handle = model.g.register_forward_hook(record)
    try:
        model.forward_with_trace(torch.randn(1, 1, 8), torch.tensor([1]))
        singleton_rows = list(logical_rows)
        logical_rows.clear()
        _pred, traces = model.forward_with_trace(torch.randn(1, 4, 8), torch.tensor([4]))
        recursive_rows = list(logical_rows)
    finally:
        handle.remove()

    assert singleton_rows == [(1, False)]
    assert sum(count for count, _is_merge in recursive_rows) == 7
    assert recursive_rows[0] == (4, False)
    assert all(is_merge for _count, is_merge in recursive_rows[1:])
    assert tuple(traces[0].shape) == (7, 8)


@pytest.mark.parametrize("operator_kind", ["fno", "tfno", "uno", "conv1d"])
def test_leaf_and_merge_losses_reach_the_same_parameter_objects(operator_kind: str) -> None:
    torch = pytest.importorskip("torch")
    family_name = "fno" if operator_kind == "fno" else "neural_operator"
    family = resolve_family(family_name, _config(operator_kind))
    family._ensure_model(output_dim=1)
    g = family._model.g
    torch.nn.init.normal_(g.residual_head.weight, std=0.1)

    left = torch.randn(3, 8)
    right = torch.randn(3, 8)
    g.zero_grad(set_to_none=True)
    g(left).square().mean().backward()
    leaf_grad_ids = {id(parameter) for parameter in g.parameters() if parameter.grad is not None}
    g.zero_grad(set_to_none=True)
    g(left, right).square().mean().backward()
    merge_grad_ids = {id(parameter) for parameter in g.parameters() if parameter.grad is not None}

    all_ids = {id(parameter) for parameter in g.parameters()}
    assert leaf_grad_ids == merge_grad_ids == all_ids


def test_split_g_checkpoint_is_rejected_instead_of_partially_loaded(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "old_split_g.pt"
    torch.save({"leaf_fno.fake": torch.zeros(1), "merge_fno.fake": torch.zeros(1)}, path)
    family = resolve_family("fno", _config("fno"))

    with pytest.raises(ValueError, match="Split leaf/merge checkpoints cannot be loaded"):
        family._maybe_warmstart(
            {
                "weights_path": str(path),
                "operator_kind": "fno",
                "output_dim": 1,
            }
        )
