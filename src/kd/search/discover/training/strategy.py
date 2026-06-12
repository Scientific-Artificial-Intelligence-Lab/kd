
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

from kd.search.discover.core.batch import Batch

if TYPE_CHECKING:
    from kd.search.discover.engine import Generator

DEFAULT_EPSILON = 0.05
DEFAULT_BASELINE = "R_e"
DEFAULT_ENTROPY_WEIGHT = 0.005
DEFAULT_GAMMA = 0.5
DEFAULT_ENTROPY_GAMMA = 1.0
DEFAULT_LEARNING_RATE = 0.001
REWARD_CLIP_ABS = 1e6
GRAD_NORM_ORDER = 2
ALLOWED_BASELINES = frozenset({"R_e", "ewma_R", "combined"})
EMPTY_LOSS_VALUE = 0.0
DEGENERATE_FILTER_THRESHOLD = 1

BoolArray = npt.NDArray[np.bool_]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineState:

    ewma_reward: float = 0.0
    n_updates: int = 0


class RSPGStrategy:

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        baseline: str = DEFAULT_BASELINE,
        entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
        gamma: float = DEFAULT_GAMMA,
        entropy_gamma: float = DEFAULT_ENTROPY_GAMMA,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        self._validate_config(epsilon, baseline, entropy_weight, gamma)
        self.epsilon = epsilon
        self.baseline = baseline
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.entropy_gamma = entropy_gamma
        self._learning_rate = learning_rate
        self._optimizer: Optimizer | None = None
        self._controller: Generator | None = None
        self._pending_optimizer_state: dict[str, Any] | None = None



        self._degenerate_filter_warned: bool = False
        self._grad_norm_warned: bool = False

    @property
    def optimizer_state(self) -> dict[str, Any] | None:
        if self._optimizer is None:
            return self._pending_optimizer_state
        state = self._optimizer.state_dict()

        cpu_state: dict[str, Any] = {"param_groups": state["param_groups"]}
        cpu_param_state: dict[int, dict[str, Any]] = {}
        for param_idx, param_state in state["state"].items():
            cpu_entry: dict[str, Any] = {}
            for key, val in param_state.items():
                if isinstance(val, torch.Tensor):
                    cpu_entry[key] = val.detach().cpu().clone()
                else:
                    cpu_entry[key] = val
            cpu_param_state[param_idx] = cpu_entry
        cpu_state["state"] = cpu_param_state
        return cpu_state

    @optimizer_state.setter
    def optimizer_state(self, state: dict[str, Any]) -> None:
        if self._optimizer is not None:
            self._optimizer.load_state_dict(state)
            self._pending_optimizer_state = None
        else:

            self._pending_optimizer_state = state

    def reset_optimizer(self) -> None:
        self._optimizer = None
        self._pending_optimizer_state = None

    def train_step(
        self,
        controller: Generator,
        batch: Batch,
        rewards: Tensor,
        baseline_state: BaselineState,
        valid_mask: BoolArray | None = None,
    ) -> tuple[dict[str, float], BaselineState]:
        self._validate_inputs(batch, rewards, baseline_state)





        filtered_batch = batch
        filtered_rewards = rewards
        if valid_mask is not None:
            filtered_batch, filtered_rewards = self._apply_valid_mask(
                batch, rewards, valid_mask,
            )
        if filtered_rewards.shape[0] == 0:
            logger.warning(
                "No valid samples remain after invalid-expression filtering; "
                "skipping update.",
            )
            return self._empty_loss_info(), baseline_state
        filtered_batch, filtered_rewards, quantile = self._filter_batch(
            filtered_batch,
            filtered_rewards,
        )
        clipped_rewards = self._clip_rewards(filtered_rewards, controller.device)
        baseline_value, new_state = self._compute_baseline(
            clipped_rewards, quantile, baseline_state,
        )
        optimizer = self._get_optimizer(controller)
        controller.train()
        optimizer.zero_grad()
        neglogp, entropy = controller.make_neglogp_and_entropy(
            filtered_batch, entropy_gamma=self.entropy_gamma,
        )
        pg_loss = self._policy_gradient_loss(clipped_rewards, baseline_value, neglogp)
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        total_loss = pg_loss + entropy_loss
        total_loss.backward()
        grad_norm = self._grad_norm(controller)
        optimizer.step()
        loss_info = self._make_loss_info(
            pg_loss,
            entropy_loss,
            total_loss,
            baseline_value,
            clipped_rewards,
            grad_norm,
        )
        return loss_info, new_state

    def _apply_valid_mask(
        self,
        batch: Batch,
        rewards: Tensor,
        valid_mask: BoolArray,
    ) -> tuple[Batch, Tensor]:
        mask = np.asarray(valid_mask, dtype=np.bool_)
        if mask.ndim != 1 or mask.shape[0] != batch.actions.shape[0]:
            raise ValueError("valid_mask must have shape (B,).")
        mask_tensor = torch.from_numpy(mask).to(device=rewards.device)
        filtered_batch = self._slice_batch(batch, mask)
        filtered_rewards = rewards[mask_tensor]
        return filtered_batch, filtered_rewards

    @staticmethod
    def _validate_config(
        epsilon: float,
        baseline: str,
        entropy_weight: float,
        gamma: float,
    ) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be in the interval (0, 1].")
        if baseline not in ALLOWED_BASELINES:
            raise ValueError(f"baseline must be one of {sorted(ALLOWED_BASELINES)}.")
        if entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the interval [0, 1].")

    @staticmethod
    def _validate_inputs(
        batch: Batch,
        rewards: Tensor,
        baseline_state: BaselineState,
    ) -> None:
        if rewards.ndim != 1:
            raise ValueError("rewards must have shape (B,).")
        if rewards.shape[0] != batch.actions.shape[0]:
            raise ValueError("rewards must align with batch size.")
        if not isinstance(baseline_state, BaselineState):
            raise TypeError("baseline_state must be a BaselineState instance.")
        if not torch.isfinite(rewards).all():
            raise ValueError("rewards must be finite (no NaN or Inf).")

    def _filter_batch(
        self,
        batch: Batch,
        rewards: Tensor,
    ) -> tuple[Batch, Tensor, float]:
        reward_values = rewards.detach().to(dtype=torch.float32, device="cpu")
        reward_np = reward_values.numpy()
        quantile = float(np.quantile(reward_np, 1.0 - self.epsilon, method="higher"))
        keep = reward_np >= quantile
        keep_tensor = torch.from_numpy(keep)
        filtered_batch = self._slice_batch(batch, keep)
        filtered_rewards = reward_values[keep_tensor]
        kept = int(filtered_rewards.shape[0])
        if kept <= DEGENERATE_FILTER_THRESHOLD and not self._degenerate_filter_warned:





            logger.warning(
                "Quantile filter kept %d sample(s) out of %d (epsilon=%.4f); "
                "policy-gradient update runs on degenerate support. "
                "Further occurrences in this strategy instance will not be "
                "re-warned.",
                kept,
                int(reward_values.shape[0]),
                self.epsilon,
            )
            self._degenerate_filter_warned = True
        return filtered_batch, filtered_rewards, quantile

    @staticmethod
    def _slice_batch(batch: Batch, keep: BoolArray) -> Batch:
        return Batch(
            actions=batch.actions[keep],
            obs=batch.obs[keep],
            priors=batch.priors[keep],
            lengths=batch.lengths[keep],
        )

    @staticmethod
    def _clip_rewards(rewards: Tensor, device: torch.device) -> Tensor:
        clipped = torch.clamp(rewards, min=-REWARD_CLIP_ABS, max=REWARD_CLIP_ABS)
        return clipped.to(device=device, dtype=torch.float32)

    @staticmethod
    def _empty_loss_info() -> dict[str, float]:
        return {
            "pg_loss": EMPTY_LOSS_VALUE,
            "entropy_loss": EMPTY_LOSS_VALUE,
            "total_loss": EMPTY_LOSS_VALUE,
            "baseline": EMPTY_LOSS_VALUE,
            "reward": EMPTY_LOSS_VALUE,
            "grad_norm": EMPTY_LOSS_VALUE,
        }

    def _compute_baseline(
        self,
        rewards: Tensor,
        quantile: float,
        baseline_state: BaselineState,
    ) -> tuple[float, BaselineState]:
        if self.baseline == "R_e":
            new_state = BaselineState(
                ewma_reward=baseline_state.ewma_reward,
                n_updates=baseline_state.n_updates + 1,
            )
            return quantile, new_state
        mean_reward = float(torch.mean(rewards).item())
        ewma_reward = self._update_ewma(mean_reward, quantile, baseline_state)
        baseline_value = (
            ewma_reward
            if self.baseline == "ewma_R"
            else quantile + ewma_reward
        )
        new_state = BaselineState(
            ewma_reward=ewma_reward,
            n_updates=baseline_state.n_updates + 1,
        )
        return baseline_value, new_state

    def _update_ewma(
        self,
        mean_reward: float,
        quantile: float,
        baseline_state: BaselineState,
    ) -> float:
        target = mean_reward if self.baseline == "ewma_R" else mean_reward - quantile
        if baseline_state.n_updates == 0:
            return target
        new_weight = 1.0 - self.gamma
        return new_weight * target + self.gamma * baseline_state.ewma_reward

    def _get_optimizer(self, controller: Generator) -> Optimizer:
        if self._optimizer is None:
            self._controller = controller
            self._optimizer = Adam(controller.parameters(), lr=self._learning_rate)
            if self._pending_optimizer_state is not None:
                self._optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
            return self._optimizer
        if controller is not self._controller:
            raise ValueError("RSPGStrategy optimizer is bound to a single controller.")
        return self._optimizer

    @staticmethod
    def _policy_gradient_loss(
        rewards: Tensor,
        baseline: float,
        neglogp: Tensor,
    ) -> Tensor:
        baseline_tensor = torch.as_tensor(
            baseline, dtype=torch.float32, device=neglogp.device,
        )
        return torch.mean((rewards - baseline_tensor) * neglogp)

    def _grad_norm(self, controller: Generator) -> float:
        total_norm = 0.0
        any_grad = False
        for parameter in controller.parameters():
            if parameter.grad is None:
                continue
            any_grad = True
            parameter_norm = parameter.grad.data.norm(GRAD_NORM_ORDER)
            total_norm += float(parameter_norm.item()) ** GRAD_NORM_ORDER
        if not any_grad and not self._grad_norm_warned:




            logger.debug(
                "_grad_norm: every controller parameter has grad=None; "
                "returning 0.0 (no backward signal reached the controller). "
                "Further occurrences in this strategy instance will be "
                "silent.",
            )
            self._grad_norm_warned = True
        return float(total_norm ** (1.0 / GRAD_NORM_ORDER))

    @staticmethod
    def _make_loss_info(
        pg_loss: Tensor,
        entropy_loss: Tensor,
        total_loss: Tensor,
        baseline: float,
        rewards: Tensor,
        grad_norm: float,
    ) -> dict[str, float]:
        return {
            "pg_loss": float(pg_loss.detach().cpu().item()),
            "entropy_loss": float(entropy_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "baseline": float(baseline),
            "reward": float(torch.mean(rewards).detach().cpu().item()),
            "grad_norm": float(grad_norm),
        }
