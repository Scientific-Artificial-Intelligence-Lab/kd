
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        inp = torch.stack([x, t], dim=-1)
        out: Tensor = self.net(inp).squeeze(-1)
        return out


def train_test_nn(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    u_target: Tensor,
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> float:
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    x_train = x.detach()
    t_train = t.detach()

    model.train()
    final_loss = float("inf")

    for _epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(x=x_train, t=t_train)
        loss = loss_fn(u_pred, u_target.detach())
        loss.backward()
        optimizer.step()
        final_loss = loss.detach().item()

    model.eval()
    return final_loss
