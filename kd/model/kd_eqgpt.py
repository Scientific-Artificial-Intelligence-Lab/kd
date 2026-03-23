"""KD_EqGPT: Knowledge-guided GPT wrapper for PDE discovery.

Wraps the EqGPT pre-trained model (GPT + surrogate NN) to perform
RL-guided equation search on wave-breaking data.
"""

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Paths resolved lazily but defined at module level (no heavy imports)
_EQGPT_DIR = Path(__file__).resolve().parent / "eqgpt"
_REF_LIB_DIR = _EQGPT_DIR.resolve().parent.parent / "ref_lib" / "EqGPT_wave_breaking"

# Surrogate training constants
_CHOOSE = 95
_NOISE_LEVEL = 0
_EQUATION_NAME = "wave_breaking"
_VARIABLES = ["t", "x"]


class KD_EqGPT:
    """Wrapper for EqGPT pre-trained PDE discovery.

    Parameters
    ----------
    pretrained : str
        Name of the pre-trained scenario. Currently only "wave_breaking".
    optimize_epochs : int
        Number of RL search epochs.
    samples_per_epoch : int
        Number of candidate equations sampled per epoch.
    case_filter : str
        "N" for N-type cases only (12 cases), "all" for all 23 cases.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pretrained: str = "wave_breaking",
        optimize_epochs: int = 5,
        samples_per_epoch: int = 400,
        case_filter: str = "N",
        seed: int = 0,
    ) -> None:
        _SUPPORTED_PRETRAINED = ("wave_breaking",)
        if pretrained not in _SUPPORTED_PRETRAINED:
            raise ValueError(
                f"pretrained must be one of {_SUPPORTED_PRETRAINED}, "
                f"got {pretrained!r}"
            )
        if not isinstance(optimize_epochs, int):
            raise TypeError(
                f"optimize_epochs must be int, got {type(optimize_epochs).__name__}"
            )
        if optimize_epochs < 1:
            raise ValueError(
                f"optimize_epochs must be >= 1, got {optimize_epochs}"
            )
        if not isinstance(samples_per_epoch, int):
            raise TypeError(
                f"samples_per_epoch must be int, "
                f"got {type(samples_per_epoch).__name__}"
            )
        if samples_per_epoch < 1:
            raise ValueError(
                f"samples_per_epoch must be >= 1, got {samples_per_epoch}"
            )
        if case_filter not in ("N", "all"):
            raise ValueError(
                f"case_filter must be 'N' or 'all', got {case_filter!r}"
            )
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.pretrained = pretrained
        self.optimize_epochs = optimize_epochs
        self.samples_per_epoch = samples_per_epoch
        self.case_filter = case_filter
        self.seed = seed

    def fit_pretrained(self) -> dict:
        """Run pre-trained GPT + surrogate RL search on wave-breaking data.

        Returns
        -------
        dict
            Keys: equations (List[str]), rewards (List[float]),
                  best_equation (str), best_reward (float).
        """
        # --- Lazy imports (avoid torch/Julia native conflicts) ---
        import pickle
        import random

        import torch
        import torch.utils.data as Data
        from torch import nn, optim

        from .eqgpt._device import device, DEVICE_STR, load_checkpoint
        from .eqgpt.gpt_model import GPT, max_pos, CLIP, word2id, id2word
        from .eqgpt.neural_network import NN
        from .eqgpt.calculate_terms import calculate_terms
        from .eqgpt.continue_train_GPT_all import (
            calculate_reward,
            get_mask_invalid,
            find_min_no_repeat,
            delete_duplicate,
            train_step,
            MyDataSet,
        )

        # --- Seed ---
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- Load pretrained GPT ---
        model_Q = GPT().to(device)
        gpt_path = _EQGPT_DIR / f"gpt_model/PDEGPT_{_EQUATION_NAME}.pt"
        model_Q.load_state_dict(load_checkpoint(str(gpt_path)))

        # --- Load wave-breaking data and build surrogates ---
        data_dict = pickle.load(
            open(str(_REF_LIB_DIR / "wave_breaking_data.pkl"), "rb")
        )
        all_Net, all_database, all_nx, all_nt = self._load_surrogates(
            data_dict, device, DEVICE_STR, load_checkpoint
        )

        # --- RL search loop ---
        words2value: Dict[str, Any] = {}
        mask_invalid = get_mask_invalid(_VARIABLES)
        best_award: Optional[List[float]] = None
        best_sentence: Optional[List[List[int]]] = None

        optimizer = optim.Adam(model_Q.parameters(), lr=1e-5)

        for epoch in range(self.optimize_epochs):
            all_reward = torch.zeros([self.samples_per_epoch])
            all_sentence: List[List[int]] = []

            for i in range(self.samples_per_epoch):
                sentence = [word2id["S"], word2id["ut"], word2id["+"]]
                while len(sentence) < max_pos - 1:
                    next_step, _prob = model_Q.step(sentence, mask_invalid)
                    sentence.append(next_step)
                    if next_step == word2id["E"]:
                        break
                # Remove start symbol before reward calculation
                sentence.pop(0)
                reward, words2value, mask_invalid = calculate_reward(
                    sentence, all_Net, all_database,
                    words2value, mask_invalid, _VARIABLES,
                    all_nx, all_nt,
                )
                all_reward[i] = reward
                all_sentence.append(sentence)

            # Update top-10 list
            best_award, best_sentence = self._update_top10(
                epoch, all_reward, all_sentence,
                best_award, best_sentence,
                find_min_no_repeat,
            )

            # Fine-tune GPT on top candidates
            continue_train_data = self._prepare_train_data(
                best_sentence, best_award, word2id, id2word, delete_duplicate,
            )
            batch_size = len(continue_train_data)
            dataset = MyDataSet(continue_train_data)
            data_loader = Data.DataLoader(
                dataset, batch_size=batch_size,
                collate_fn=dataset.padding_batch,
            )
            criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
            for _ in range(5):
                train_step(model_Q, data_loader, optimizer, criterion, CLIP)

        # --- Build result ---
        equations, rewards = self._format_results(
            best_sentence, best_award, id2word, delete_duplicate,
        )
        return {
            "equations": equations,
            "rewards": rewards,
            "best_equation": equations[0] if equations else "",
            "best_reward": rewards[0] if rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_surrogates(
        self,
        data_dict: dict,
        device: Any,
        device_str: str,
        load_checkpoint: Any,
    ) -> Tuple[list, list, list, list]:
        """Build surrogate NN models for each case (replaces get_meta)."""
        import torch
        from .eqgpt.neural_network import NN

        all_Net: List[Any] = []
        all_database: List[Any] = []
        all_nx: List[int] = []
        all_nt: List[int] = []

        for name in sorted(data_dict.keys()):
            if self.case_filter == "N" and "N" not in name:
                continue
            data = data_dict[name]
            net, db, nx, nt = self._build_single_surrogate(
                name, data, device, device_str, load_checkpoint,
            )
            all_Net.append(net)
            all_database.append(db)
            all_nx.append(nx)
            all_nt.append(nt)

        return all_Net, all_database, all_nx, all_nt

    def _build_single_surrogate(
        self,
        trail_num: str,
        data: np.ndarray,
        device: Any,
        device_str: str,
        load_checkpoint: Any,
    ) -> Tuple[Any, Any, int, int]:
        """Build one surrogate NN + database for a single case."""
        import torch
        from .eqgpt.neural_network import NN

        net = NN(
            Num_Hidden_Layers=6,
            Neurons_Per_Layer=60,
            Input_Dim=2,
            Output_Dim=1,
            Data_Type=torch.float32,
            Device=device_str,
            Activation_Function="Sin",
            Batch_Norm=False,
        )

        model_dir = (
            _EQGPT_DIR
            / f"model_save/{_EQUATION_NAME}"
            / f"{_CHOOSE}_{_NOISE_LEVEL}_{trail_num}(Non_unit)"
        )
        best_epoch = int(np.load(str(model_dir / "best_epoch.npy"))[0])
        load_state = f"Net_Sin_{best_epoch}"
        net.load_state_dict(
            load_checkpoint(str(model_dir / f"{load_state}.pkl"))
        )
        net.eval()

        # Parse physical parameters from case name
        pattern = r"G(\d+)Tp(\d+)A(\d+)"
        match = re.search(pattern, trail_num)
        if match is None:
            raise ValueError(f"Cannot parse case name: {trail_num}")
        g, tp_int, _a = map(int, match.groups())
        tp = tp_int / 10.0
        lamda = 9.81 * (tp ** 2) / (2.0 * math.pi)

        # Build coordinate grid
        x = np.concatenate([
            np.linspace(8.18, 9.34, 50),
            np.linspace(9.77, 10.93, 50),
            np.linspace(11.41, 12.57, 50),
        ])
        t = np.arange(
            0.05 + 0.1, np.max(data[:, 0]) - 0.1, 0.05
        )
        x = (x - 8.17) / lamda
        t = t / tp

        nx = x.shape[0]
        nt = t.shape[0]
        T, X = np.meshgrid(t, x, indexing="ij")
        inputs = np.stack([T.ravel(), X.ravel()], axis=1)
        inputs_tensor = torch.from_numpy(
            inputs.astype(np.float32)
        ).to(device)
        database = torch.tensor(inputs_tensor, requires_grad=True)

        return net, database, nx, nt

    @staticmethod
    def _update_top10(
        epoch: int,
        all_reward: Any,
        all_sentence: List[List[int]],
        best_award: Optional[List[float]],
        best_sentence: Optional[List[List[int]]],
        find_min_no_repeat: Any,
    ) -> Tuple[List[float], List[List[int]]]:
        """Merge current epoch candidates into running top-10."""
        if epoch == 0 or best_award is None or best_sentence is None:
            best_index, best_award_new = find_min_no_repeat(all_reward)
            best_sentence_new = [all_sentence[idx] for idx in best_index]
            return best_award_new, best_sentence_new

        second_index, second_award = find_min_no_repeat(all_reward)
        for p_idx in range(len(second_award)):
            potential_award = second_award[p_idx]
            potential_text = all_sentence[second_index[p_idx]]
            array = np.asarray(best_award)
            idx_arr = np.where((array - potential_award) < 0)[0]
            if len(idx_arr) == 0:
                continue
            if np.isin(potential_award, array):
                continue
            ins_idx = idx_arr[0]
            best_sentence.insert(ins_idx, potential_text)
            best_award.insert(ins_idx, potential_award)
            best_sentence.pop(-1)
            best_award.pop(-1)

        return best_award, best_sentence

    @staticmethod
    def _prepare_train_data(
        best_sentence: List[List[int]],
        best_award: List[float],
        word2id: dict,
        id2word: list,
        delete_duplicate: Any,
    ) -> List[List[int]]:
        """Format top sentences into training data for GPT fine-tuning."""
        continue_train_data: List[List[int]] = []
        for i in range(len(best_award)):
            sentence_data = list(best_sentence[i])
            sentence_data = delete_duplicate(sentence_data)
            if sentence_data[-1] == word2id["E"]:
                sentence_data.pop(-1)
            if word2id["S"] not in sentence_data:
                sentence_data.insert(0, word2id["S"])
                sentence_data.append(word2id["E"])
            continue_train_data.append(sentence_data)
            vis = [id2word[int(tok)] for tok in sentence_data]
            logger.info(
                "Top-%d: %s  reward=%.4f",
                i + 1, "".join(vis[1:-1]), best_award[i],
            )
        return continue_train_data

    @staticmethod
    def _format_results(
        best_sentence: Optional[List[List[int]]],
        best_award: Optional[List[float]],
        id2word: list,
        delete_duplicate: Any,
    ) -> Tuple[List[str], List[float]]:
        """Convert final top-10 into human-readable equations."""
        if best_sentence is None or best_award is None:
            return [], []
        equations: List[str] = []
        rewards: List[float] = []
        for i in range(len(best_award)):
            sentence_data = list(best_sentence[i])
            sentence_data = delete_duplicate(sentence_data)
            if sentence_data[-1] == 1:  # 'E' token
                sentence_data.pop(-1)
            if 5 not in sentence_data:  # 'S' token
                sentence_data.insert(0, 5)
                sentence_data.append(1)
            vis = [id2word[int(tok)] for tok in sentence_data]
            eq_str = "".join(vis[1:-1])
            equations.append(eq_str)
            rewards.append(float(best_award[i]))
        return equations, rewards
