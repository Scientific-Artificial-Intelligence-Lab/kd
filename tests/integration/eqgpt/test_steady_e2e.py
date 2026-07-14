
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.core.equation import Equation, Form, Scalar
from kd.core.platform.builder import PlatformBuilder
from kd.data.schema import PDEDataset
from kd.search.eqgpt.backend import FakeGPTBackend
from kd.search.eqgpt.config import EqGPTConfig
from kd.search.eqgpt.plugin import EqGPTPlugin
from kd.search.eqgpt.vocab import E_ID, PLUS_ID, S_ID, load_vocab
from kd.search.result import ExperimentResult
from kd.search.runner import ExperimentRunner

pytestmark = pytest.mark.integration


class ScriptedGPTBackend(FakeGPTBackend):

    def __init__(
        self, vocab_size: int, target: list[int], *, start_len: int = 1, seed: int = 0
    ) -> None:
        super().__init__(vocab_size, seed=seed)
        self._target = list(target)
        self._start_len = start_len

    def next_token_logits(self, prefix: list[int]) -> Tensor:
        token = self._target[len(prefix) - self._start_len]
        logits = torch.full((self.vocab_size,), -1e9)
        logits[token] = 1e9
        return logits


def _harmonic_scatter_dataset() -> PDEDataset:
    torch.manual_seed(0)
    x = torch.rand(16, dtype=torch.float64)
    y = torch.rand(16, dtype=torch.float64)
    u = x**3 - 3.0 * x * y**2
    return PDEDataset.from_scatter(
        coords={"x": x, "y": y}, fields={"u": u}, lhs="", name="steady_harmonic"
    )


def _scripted_steady_plugin() -> EqGPTPlugin:
    vocab = load_vocab()
    target = [vocab.word2id["uxx"], PLUS_ID, vocab.word2id["uyy"], E_ID]
    backend = ScriptedGPTBackend(vocab.size, target, start_len=len((S_ID,)))

    config = EqGPTConfig(
        sparsity_alpha=1.0,
        steady=True,
        steady_activation="sin",
        start_words=("S",),
        steady_train_iters=20,
        samples_per_epoch=4,
        top_k=2,
        exploration_rate=0.0,
        max_length=12,
    )
    return EqGPTPlugin(config, backend=backend)


def _run_steady_discovery() -> ExperimentResult:
    dataset = _harmonic_scatter_dataset()
    plugin = _scripted_steady_plugin()

    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    return ExperimentRunner(plugin, max_iterations=1, batch_size=4).run(components)


def test_steady_fake_backend_discovers_homogeneous_structure() -> None:
    result = _run_steady_discovery()


    assert result.equation is not None
    assert isinstance(result.equation, Equation)
    assert result.equation.form is Form.HOMOGENEOUS



    term_irs = [ir for ir, _c in result.equation.terms]
    assert term_irs == ["u_xx", "u_yy"]
    assert result.equation.terms[0][1] == Scalar(1.0)



    final = result.final_eval
    assert final.form is Form.HOMOGENEOUS
    assert final.is_valid is True
    assert final.terms == ["u_xx", "u_yy"]
    assert final.coefficients is not None
    assert len(final.coefficients) == len(final.terms)
