from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class TorchMSEProxyOracle:
    """Minimal torch-based proxy oracle for scalar regression with MSE loss.

    This is intentionally lightweight: it is a convenience wrapper for small
    proxy models used in simulations or quick experiments.
    """

    model: Any
    lr: float = 1e-3
    n_epochs: int = 10
    batch_size: int = 32
    device: Optional[str] = None

    def fit(
        self,
        inputs: Sequence[Any],
        targets: Sequence[float],
        *,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "TorchMSEProxyOracle":  # pragma: no cover
        if torch is None or nn is None:
            raise ImportError("torch is required for TorchMSEProxyOracle. Install with: uv sync --extra torch")

        device = torch.device(self.device) if self.device else torch.device("cpu")
        model = self.model.to(device)
        model.train()

        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        y = torch.as_tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
        if sample_weight is None:
            w = None
        else:
            w = torch.as_tensor(sample_weight, dtype=torch.float32, device=device).view(-1, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr))
        mse = nn.MSELoss(reduction="none")

        n = x.shape[0]
        bs = max(1, int(self.batch_size))
        for _ in range(max(1, int(self.n_epochs))):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                pred = model(x[idx])
                loss_vec = mse(pred, y[idx])
                if w is not None:
                    loss_vec = loss_vec * w[idx]
                loss = loss_vec.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, inputs: Sequence[Any]) -> Any:  # pragma: no cover
        if torch is None:
            raise ImportError("torch is required for TorchMSEProxyOracle. Install with: uv sync --extra torch")
        device = torch.device(self.device) if self.device else torch.device("cpu")
        model = self.model.to(device)
        model.eval()
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(x)
        return pred.detach().cpu().numpy()

