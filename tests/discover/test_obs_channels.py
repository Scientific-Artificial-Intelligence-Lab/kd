
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.search.discover.builder import (
    build_controller,
    build_library,
    build_prior_system,
)
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.controller.lstm import LSTMController
from kd.search.discover.core.batch import Batch
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)





BURGERS_CONFIG = LibraryConfig(
    operators=["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"],
    state_vars=["u"],
    coord_vars=["x", "t"],
)




N_TOKENS = 11
N_ACTION_INPUTS = 12
N_PARENT_INPUTS = 9
N_SIBLING_INPUTS = 12

MAX_LENGTH = 30
MIN_LENGTH = 4
BATCH_SIZE = 16
SEED = 42
EMBEDDING_DIM = 8







@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def prior_system(lib: Library) -> PriorSystem:
    return PriorSystem(lib, [
        LengthConstraint(lib, min_=MIN_LENGTH, max_=MAX_LENGTH),
        DiffChildConstraint(lib),
    ])


def _make_controller(
    lib: Library,
    prior_system: PriorSystem,
    *,
    observe_parent: bool = True,
    observe_sibling: bool = True,
    observe_action: bool = False,
    observe_dangling: bool = False,
    use_embedding: bool = False,
    embedding_dim: int = EMBEDDING_DIM,
) -> LSTMController:
    return LSTMController(
        library=lib,
        prior_system=prior_system,
        num_units=32,
        num_layers=1,
        embedding_dim=embedding_dim,
        observe_parent=observe_parent,
        observe_sibling=observe_sibling,
        observe_action=observe_action,
        observe_dangling=observe_dangling,
        use_embedding=use_embedding,
    )







class TestLibraryDimensions:

    @pytest.mark.unit
    def test_library_dimensions(self, lib: Library) -> None:
        assert len(lib.tokens) == N_TOKENS
        assert lib.n_action_inputs == N_ACTION_INPUTS
        assert lib.n_parent_inputs == N_PARENT_INPUTS
        assert lib.n_sibling_inputs == N_SIBLING_INPUTS







class TestConfigDefaults:

    @pytest.mark.unit
    def test_observe_parent_default_true(self) -> None:
        config = DiscoverConfig()
        assert config.observe_parent is True

    @pytest.mark.unit
    def test_observe_sibling_default_true(self) -> None:
        config = DiscoverConfig()
        assert config.observe_sibling is True

    @pytest.mark.unit
    def test_observe_action_default_false(self) -> None:
        config = DiscoverConfig()
        assert config.observe_action is False

    @pytest.mark.unit
    def test_observe_dangling_default_false(self) -> None:
        config = DiscoverConfig()
        assert config.observe_dangling is False

    @pytest.mark.unit
    def test_use_embedding_default_false(self) -> None:
        config = DiscoverConfig()
        assert config.use_embedding is False

    @pytest.mark.unit
    def test_config_is_frozen(self) -> None:
        config = DiscoverConfig()
        with pytest.raises(AttributeError):
            config.observe_parent = False

    @pytest.mark.unit
    def test_all_observation_flags_configurable(self) -> None:
        config = DiscoverConfig(
            observe_parent=True,
            observe_sibling=True,
            observe_action=True,
            observe_dangling=True,
            use_embedding=True,
        )
        assert config.observe_action is True
        assert config.observe_dangling is True
        assert config.use_embedding is True







class TestControllerConstructor:

    @pytest.mark.unit
    def test_constructor_accepts_observe_flags(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:


        ctrl = _make_controller(
            lib, prior_system,
            observe_parent=True,
            observe_sibling=True,
            observe_action=False,
            observe_dangling=False,
            use_embedding=False,
        )
        assert isinstance(ctrl, LSTMController)

    @pytest.mark.unit
    def test_at_least_one_channel_required(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        with pytest.raises(ValueError, match="[Aa]t least one"):
            _make_controller(
                lib, prior_system,
                observe_parent=False,
                observe_sibling=False,
                observe_action=False,
                observe_dangling=False,
            )







class TestOneHotMode:

    @pytest.mark.unit
    def test_default_onehot_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        expected_dim = N_PARENT_INPUTS + N_SIBLING_INPUTS
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"Default one-hot input_dim: expected {expected_dim}, got {actual_dim}"
        )

    @pytest.mark.unit
    def test_no_embedding_layers_in_onehot_mode(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system, use_embedding=False)
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        assert len(embedding_modules) == 0, (
            f"One-hot mode should have 0 nn.Embedding, got {len(embedding_modules)}"
        )

    @pytest.mark.unit
    def test_all_channels_onehot_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=False,
        )
        expected_dim = (
            N_ACTION_INPUTS + N_PARENT_INPUTS + N_SIBLING_INPUTS + 1
        )
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"All-channels one-hot input_dim: expected {expected_dim}, got {actual_dim}"
        )

    @pytest.mark.unit
    def test_action_only_onehot_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=False,
            observe_sibling=False,
            observe_dangling=False,
            use_embedding=False,
        )
        expected_dim = N_ACTION_INPUTS
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim

    @pytest.mark.unit
    def test_dangling_only_not_onehot(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=False,
            observe_parent=False,
            observe_sibling=False,
            observe_dangling=True,
            use_embedding=False,
        )
        expected_dim = 1
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim

    @pytest.mark.unit
    def test_parent_sibling_dangling_onehot(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=False,
        )
        expected_dim = N_PARENT_INPUTS + N_SIBLING_INPUTS + 1
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim







class TestEmbeddingMode:

    @pytest.mark.unit
    def test_default_embedding_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            use_embedding=True,
            embedding_dim=EMBEDDING_DIM,
        )
        expected_dim = 2 * EMBEDDING_DIM
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"Default embedding input_dim: expected {expected_dim}, got {actual_dim}"
        )

    @pytest.mark.unit
    def test_embedding_layers_created(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            use_embedding=True,
        )
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        assert len(embedding_modules) == 2, (
            f"Expected 2 nn.Embedding (parent+sibling), got {len(embedding_modules)}"
        )

    @pytest.mark.unit
    def test_all_channels_embedding_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
            embedding_dim=EMBEDDING_DIM,
        )
        expected_dim = 3 * EMBEDDING_DIM + 1
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"All-channels embedding input_dim: expected {expected_dim}, "
            f"got {actual_dim}"
        )

    @pytest.mark.unit
    def test_all_channels_embedding_3_embeddings(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
        )
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        assert len(embedding_modules) == 3, (
            f"Expected 3 nn.Embedding (action+parent+sibling), "
            f"got {len(embedding_modules)}"
        )

    @pytest.mark.unit
    def test_embedding_vocab_sizes(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            use_embedding=True,
        )
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        vocab_sizes = sorted(m.num_embeddings for m in embedding_modules)

        expected_sizes = sorted([N_ACTION_INPUTS, N_PARENT_INPUTS, N_SIBLING_INPUTS])
        assert vocab_sizes == expected_sizes, (
            f"Embedding vocab sizes: expected {expected_sizes}, got {vocab_sizes}"
        )

    @pytest.mark.unit
    def test_action_only_embedding_input_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=False,
            observe_sibling=False,
            observe_dangling=False,
            use_embedding=True,
            embedding_dim=EMBEDDING_DIM,
        )
        expected_dim = EMBEDDING_DIM
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim







class TestSampleWithObsConfig:

    @pytest.mark.smoke
    def test_sample_default_config(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        assert isinstance(batch, Batch)
        assert batch.actions.shape[0] == BATCH_SIZE
        assert batch.actions.dtype == np.int32
        assert np.all(batch.lengths >= 1)
        assert np.all(batch.lengths <= MAX_LENGTH)

    @pytest.mark.smoke
    def test_sample_all_channels_embedding(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        assert isinstance(batch, Batch)
        assert batch.actions.shape[0] == BATCH_SIZE
        assert np.all(batch.lengths >= 1)

    @pytest.mark.unit
    def test_sample_expressions_valid_default(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            tokens = batch.actions[i, :length].tolist()
            dangling = 1
            for token_idx in tokens:
                dangling += lib[token_idx].arity - 1
            assert dangling == 0, (
                f"Seq {i} has dangling={dangling} at length={length}"
            )

    @pytest.mark.unit
    def test_sample_expressions_valid_all_channels(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            length = int(batch.lengths[i])
            tokens = batch.actions[i, :length].tolist()
            dangling = 1
            for token_idx in tokens:
                dangling += lib[token_idx].arity - 1
            assert dangling == 0, (
                f"Seq {i} has dangling={dangling} at length={length}"
            )

    @pytest.mark.unit
    def test_obs_shape_unchanged(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        B, L = batch.actions.shape
        assert batch.obs.shape == (B, 4, L), (
            f"obs shape must be (B, 4, L), got {batch.obs.shape}"
        )







class TestNeglogpWithObsConfig:

    @pytest.mark.unit
    def test_neglogp_default_config(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        neglogp, entropy = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        assert neglogp.shape == (BATCH_SIZE,)
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.all(torch.isfinite(neglogp))
        assert torch.all(torch.isfinite(entropy))

    @pytest.mark.unit
    def test_neglogp_all_channels_embedding(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        neglogp, entropy = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        assert neglogp.shape == (BATCH_SIZE,)
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.all(torch.isfinite(neglogp))
        assert torch.all(torch.isfinite(entropy))

    @pytest.mark.unit
    def test_gradients_flow_default(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        ctrl.zero_grad()
        neglogp, _ = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        neglogp.mean().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in ctrl.parameters()
        )
        assert has_grad, "No gradients flowed in default one-hot mode"

    @pytest.mark.unit
    def test_gradients_flow_embedding(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(
            lib, prior_system,
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
        )
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        ctrl.zero_grad()
        neglogp, _ = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        neglogp.mean().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in ctrl.parameters()
        )
        assert has_grad, "No gradients flowed in embedding mode"

    @pytest.mark.unit
    def test_neglogp_positive_default(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl = _make_controller(lib, prior_system)
        torch.manual_seed(SEED)
        batch = ctrl.sample(BATCH_SIZE)
        neglogp, _ = ctrl.make_neglogp_and_entropy(batch, entropy_gamma=1.0)
        assert torch.all(neglogp > 0), "neglogp must be positive"







class TestBuilderObsConfig:

    @pytest.mark.unit
    def test_builder_default_passes_reference_config(self) -> None:
        config = DiscoverConfig()
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        torch.manual_seed(SEED)
        ctrl = build_controller(lib, ps, config)
        expected_dim = lib.n_parent_inputs + lib.n_sibling_inputs
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"Builder default: expected input_dim={expected_dim}, got {actual_dim}"
        )

    @pytest.mark.unit
    def test_builder_all_channels_embedding(self) -> None:
        config = DiscoverConfig(
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            observe_dangling=True,
            use_embedding=True,
            embedding_dim=8,
        )
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        torch.manual_seed(SEED)
        ctrl = build_controller(lib, ps, config)
        expected_dim = 3 * 8 + 1
        actual_dim = ctrl.rnn.cells[0].input_size
        assert actual_dim == expected_dim, (
            f"Builder all+emb: expected input_dim={expected_dim}, got {actual_dim}"
        )

    @pytest.mark.unit
    def test_builder_passes_use_embedding(self) -> None:
        config = DiscoverConfig(
            observe_action=True,
            observe_parent=True,
            observe_sibling=True,
            use_embedding=True,
        )
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        torch.manual_seed(SEED)
        ctrl = build_controller(lib, ps, config)
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        assert len(embedding_modules) == 3, (
            f"Expected 3 nn.Embedding from builder, got {len(embedding_modules)}"
        )

    @pytest.mark.unit
    def test_builder_default_no_embeddings(self) -> None:
        config = DiscoverConfig()
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        torch.manual_seed(SEED)
        ctrl = build_controller(lib, ps, config)
        embedding_modules = [
            m for m in ctrl.modules() if isinstance(m, torch.nn.Embedding)
        ]
        assert len(embedding_modules) == 0, (
            f"Default config should have 0 nn.Embedding, got {len(embedding_modules)}"
        )

    @pytest.mark.unit
    def test_builder_functional_sample_default(self) -> None:
        config = DiscoverConfig()
        lib = build_library(config)
        ps = build_prior_system(lib, config)
        torch.manual_seed(SEED)
        ctrl = build_controller(lib, ps, config)
        batch = ctrl.sample(8)
        assert isinstance(batch, Batch)
        assert batch.actions.shape[0] == 8







class TestAntiRegression:

    @pytest.mark.unit
    def test_different_configs_different_input_dims(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl_default = _make_controller(lib, prior_system)
        ctrl_all_onehot = _make_controller(
            lib, prior_system,
            observe_action=True, observe_dangling=True,
        )
        ctrl_emb = _make_controller(
            lib, prior_system, use_embedding=True,
        )
        dim_default = ctrl_default.rnn.cells[0].input_size
        dim_all_onehot = ctrl_all_onehot.rnn.cells[0].input_size
        dim_emb = ctrl_emb.rnn.cells[0].input_size


        assert dim_default != dim_all_onehot, (
            "Default and all-channel one-hot should have different input_dim"
        )
        assert dim_default != dim_emb, (
            "Default one-hot and default embedding should have different input_dim"
        )
        assert dim_all_onehot != dim_emb, (
            "All-channel one-hot and default embedding should have different input_dim"
        )

    @pytest.mark.unit
    def test_onehot_dim_independent_of_embedding_dim(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl_a = _make_controller(
            lib, prior_system, use_embedding=False, embedding_dim=4,
        )
        ctrl_b = _make_controller(
            lib, prior_system, use_embedding=False, embedding_dim=16,
        )
        assert ctrl_a.rnn.cells[0].input_size == ctrl_b.rnn.cells[0].input_size

    @pytest.mark.unit
    def test_embedding_dim_affects_embedding_mode(
        self, lib: Library, prior_system: PriorSystem,
    ) -> None:
        ctrl_a = _make_controller(
            lib, prior_system, use_embedding=True, embedding_dim=4,
        )
        ctrl_b = _make_controller(
            lib, prior_system, use_embedding=True, embedding_dim=16,
        )
        assert ctrl_a.rnn.cells[0].input_size == 2 * 4
        assert ctrl_b.rnn.cells[0].input_size == 2 * 16
