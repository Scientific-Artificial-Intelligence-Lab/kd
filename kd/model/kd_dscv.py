"""Deep Reinforcement Learning for PDE Discovery.

This module implements a deep reinforcement learning approach for discovering
governing equations of PDEs. It supports both regular grid data and sparse
sampling, and can be combined with Physics-Informed Neural Networks (PINN).
"""

from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator
import warnings
import os
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time
from datetime import datetime
import logging
import numpy as np
import commentjson as json
import torch
from scipy.stats import pearsonr

from .discover.task import set_task
from .discover.controller import Controller
from .discover.dso.prior import make_prior
from .discover.program import Program,from_str_tokens,from_tokens
from .discover.config import load_config
from .discover.state_manager import make_state_manager as manager_make_state_manager
from .discover.pinn import PINN_model
from .discover.utils import safe_merge_dicts
from .discover.searcher import Searcher
from ..data import SparseData, RegularData

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BaseRL(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for reinforcement learning based models."""
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit model to data."""

    def predict(self, X):
        """Make predictions."""
        pass
    
class KD_DSCV(BaseRL):
    """Deep Reinforcement Learning model for PDE discovery.
    
    This class implements a deep RL approach to discover governing equations
    of PDEs. It uses a controller to generate candidate equations and a
    searcher to evaluate them.
    
    Attributes:
        n_iterations (int): Number of training iterations.
        n_samples_per_batch (int): Number of samples per batch.
        operator (list): List of allowed operators.
        out_path (str): Path for output files.
        dataset: Dataset object.
        seed (int): Random seed.
        core_num (int): Number of CPU cores to use.
        config: Configuration dictionary.
    """
    
    _parameter: dict = {}
    
    def __init__(self,
        n_iterations=100,
        n_samples_per_batch=100,
        binary_operators=[],
        unary_operators=[],
        out_path='./log/',
        core_num=1,
        config_out=None,
        seed=0,
        ):
        """Initialize KD_DSCV model.
        
        Args:
            n_iterations (int): Number of training iterations.
            n_samples_per_batch (int): Number of samples per batch.
            binary_operators (list): List of binary operators.
            unary_operators (list): List of unary operators.
            out_path (str): Path for output files.
            core_num (int): Number of CPU cores to use.
            config_out: Optional output configuration.
            seed (int): Random seed.
        """
        self.n_iterations = n_iterations
        self.n_samples_per_batch = n_samples_per_batch
        self.operator = binary_operators + unary_operators
        self.out_path = out_path
         
        self.dataset = None
        self.seed = seed
        self.core_num = core_num
        self.base_config_file = "./discover/config/config_pde.json"
        self.set_config(config_out)

    def setup(self):
        """Setup model components.
        
        Initializes:
            - Program cache
            - Task
            - Prior
            - State manager
            - Controller
            - GP aggregator
            - Searcher
        """
        # Clear the cache and reset the compute graph
        Program.clear_cache()
                
        # initialize
        self.set_task()
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.controller = self.make_controller()
        self.gp_aggregator = self.make_gp_aggregator()
        self.searcher = self.make_searcher()
        self.set_seeds(self.seed)
        
    def info(self):
        """Print model information."""
        print("Library: ", Program.task.library)
        self.controller.prior.report_constraint_counts()
        
    def import_inner_data(self, dataset, data_type='regular'):
        """Import data from predefined datasets.

        Args:
            dataset (str): Name of dataset.
            data_type (str): Type of data ('regular' or 'sparse').
        """
        if data_type == 'regular':
            self.data_class = RegularData(dataset)
            self.data_class.load_data()
            
            # Save file
            if self.out_path is not None:
                self.out_path = os.path.join(self.out_path,
                                    "discover_{}_{}.csv".format(dataset, self.seed)
                )            
        self.setup()

    def import_dataset(self, dataset, *, sym_true=None, n_input_dim=None, dataset_name=None):
        """Import data through a :class:`~kd.dataset.PDEDataset` instance.

        This keeps the legacy :meth:`import_inner_data` path untouched while
        allowing external callers to provide data via the unified
        :func:`kd.dataset.load_pde` / :class:`PDEDataset` pipeline.

        Args:
            dataset: A :class:`kd.dataset.PDEDataset` instance.
            sym_true: Optional override for the ground-truth symbolic equation.
            n_input_dim: Optional override for the number of spatial input
                dimensions.

        Returns:
            KD_DSCV: The estimator instance (for chaining).
        """

        from kd.dataset import PDEDataset  # lazy import to avoid circular dependency
        from .discover.adapter import DSCVRegularAdapter

        if not isinstance(dataset, PDEDataset):
            raise TypeError("dataset must be a PDEDataset instance")

        adapter = DSCVRegularAdapter(dataset, sym_true=sym_true, n_input_dim=n_input_dim)
        self.data_class = adapter
        self.dataset_ = dataset

        resolved_name = dataset_name or getattr(dataset, 'legacy_name', None) or \
            getattr(dataset, 'registry_name', None) or getattr(dataset, 'equation_name', None) or "custom_dataset"
        self.dataset = resolved_name
        if self.out_path is not None:
            self.out_path = os.path.join(
                self.out_path,
                f"discover_{resolved_name}_{self.seed}.csv"
            )

        self.setup()
        return self

    def make_outter_data(self, x, y, domains, data_type):
        """Create data handler for external data.
        
        Args:
            x: Input features.
            y: Target values.
            domains: Domain boundaries.
            data_type (str): Type of data.
            
        Raises:
            AssertionError: If data type is not 'regular'.
        """
        assert data_type == 'regular', "only regular form of dataset is supported in current mode"
        # Todo (KD 1.x): legacy path, kept only for upstream compatibility.
        # In KD it is recommended to use PDEDataset + import_dataset/fit_from_dataset/fit_dataset.

    def fit(self, X, y, domains=[], data_type='Sparse'):
        """Deprecated external ``(X, y)`` entry point (disabled in KD 1.0).

        Use :meth:`import_dataset`, :meth:`fit_from_dataset` or
        :meth:`fit_dataset` together with :class:`kd.dataset.PDEDataset`
        instead of calling :meth:`fit` with raw arrays.

        Users who need full control over ``X``/``y``/``domains`` should refer
        to the native DISCOVER ``PDETask`` examples.
        """
        raise RuntimeError(
            "KD_DSCV.fit(X, y, ...) is deprecated and no longer supported in KD 1.0. "
            "Please use import_dataset / fit_from_dataset / fit_dataset together "
            "with a kd.dataset.PDEDataset instance."
        )

    def fit_dataset(
        self,
        dataset,
        *,
        n_epochs: int = 100,
        verbose: bool = True,
        sym_true=None,
        n_input_dim=None,
        dataset_name=None,
    ):
        """Train DSCV (local PDE mode) directly from a :class:`PDEDataset`.

        This is a convenience wrapper aligned with :class:`KD_SGA` and
        :class:`KD_DLGA`, using the unified dataset-based entry point.
        """

        # 真实实现路径：导入 PDEDataset 并启动 searcher
        self.import_dataset(
            dataset,
            sym_true=sym_true,
            n_input_dim=n_input_dim,
            dataset_name=dataset_name,
        )
        return self.train(n_epochs=n_epochs, verbose=verbose)

    def train_one_step(self, epoch=0, verbose=True):
        """Train model for one step.
        
        Args:
            epoch (int): Current epoch.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Training results.
        """
        return self.searcher.search_one_step(epoch=epoch, verbose=verbose)

    def train(self, n_epochs=100, verbose=True):
        """Train model for multiple epochs.
        
        Args:
            n_epochs (int): Number of epochs.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Training results.
        """
        return self.searcher.search(n_epochs=n_epochs, verbose=verbose)
        
    def set_config(self, config):
        """Set model configuration.
        
        Args:
            config: Configuration dictionary or path.
        """
        if config is not None:
            config = load_config(config)

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.base_config_file), encoding='utf-8') as f:    
            base_config = json.load(f)
        config_update = safe_merge_dicts(base_config, config)
        self.config = defaultdict(dict, config_update)
        self.config_task = self.config["task"]
        # Snapshot original function_set for safe reset on re-import.
        self._base_function_set = list(self.config_task.get("function_set", []))
        self.config_prior = self.config["prior"]
        self.config_state_manager = self.config["state_manager"]
        self.config_controller = self.config["controller"]
        self.config_gp_agg = self.config["gp_agg"]
        self.config_training = self.config["training"]

    def make_prior(self):
        """Create prior for equation generation.
        
        Returns:
            Prior: Prior object.
        """
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_state_manager(self):
        """Create state manager.
        
        Returns:
            StateManager: State manager object.
        """
        return manager_make_state_manager(self.config_state_manager)

    def make_controller(self):
        """Create controller.
        
        Returns:
            Controller: Controller object.
        """
        controller = Controller(
                                self.prior,
                                self.state_manager,
                                **self.config_controller)
        return controller

    def make_gp_aggregator(self):
        """Create GP aggregator if enabled.
        
        Returns:
            GPAggregator or None: GP aggregator object.
        """
        # KD 1.x 中禁用 GP aggregator / 多进程聚合逻辑，避免 self.pool / import 路径等历史问题。
        # 即使配置中设置了 run_gp_agg=True，也只给出一次性告警并返回 None。
        run_gp_agg = bool(self.config_gp_agg.get("run_gp_agg", False))
        if run_gp_agg:
            warnings.warn(
                "KD_DSCV: GP aggregator (run_gp_agg=True) 在 KD 1.x 中不受支持，将被忽略。",
                RuntimeWarning,
            )
        return None
    
    def make_searcher(self):
        """Create searcher.
        
        Returns:
            Searcher: Searcher object.
        """
        self.config_training['n_iterations'] = self.n_iterations
        self.config_training['n_samples_per_batch'] = self.n_samples_per_batch

        searcher = Searcher(
                            controller=self.controller,
                            args=self.config_training,
                            gp_aggregator=self.gp_aggregator
                            )
        return searcher

    # Diff tokens for each spatial dimension (used for auto function_set).
    _DIFF_TOKENS_BY_DIM = {
        1: ["diff", "diff2", "diff3"],
        2: ["Diff", "Diff2", "lap"],
        3: ["Diff_3", "Diff2_3"],
    }
    # All diff-family tokens across all dimensions (used to strip before replacing).
    _ALL_DIFF_TOKENS = {
        "diff", "diff2", "diff3", "diff4",
        "Diff", "Diff2", "lap",
        "Diff_3", "Diff2_3",
    }

    def set_task(self):
        """Set task configuration and optimizer."""
        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        self.config_task['dataset'] = self.dataset
        if len(self.operator) > 0:
            self.config_task['function_set'] = self.operator
        else:
            self._auto_select_function_set()

        assert self.data_class is not None, "Dataset should be made before setting task"
        set_task(self.config_task, self.data_class.get_data())

    def _auto_select_function_set(self) -> None:
        """Replace default 1D diff tokens with dimension-appropriate ones.

        Only activates when the user did not specify ``operator`` and
        ``n_input_dim > 1``.  Preserves non-diff tokens (add, mul, etc.).

        Always resets from ``_base_function_set`` to avoid stale tokens
        when the same ``KD_DSCV`` instance is reused across datasets.
        """
        # Reset to base config snapshot to avoid stale mutations.
        self.config_task["function_set"] = list(self._base_function_set)

        if self.data_class is None:
            return
        data = self.data_class.get_data()
        n_dim = data.get("n_input_dim", 1)
        if n_dim <= 1:
            return
        if n_dim not in self._DIFF_TOKENS_BY_DIM:
            return

        # Strip ALL diff-family tokens (across all dims) to avoid duplicates,
        # then inject exactly the target-dimension tokens.
        current = list(self._base_function_set)
        non_diff = [t for t in current if t not in self._ALL_DIFF_TOKENS]
        new_diff = self._DIFF_TOKENS_BY_DIM[n_dim]
        self.config_task["function_set"] = list(dict.fromkeys(new_diff + non_diff))

    def set_seeds(self, new_seed):
        """Set random seeds for reproducibility.
        
        Args:
            new_seed (int): Random seed.
        """
        np.random.seed(new_seed)
        random.seed(new_seed)
        torch.random.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)
    
    def print_pq(self):
        """Print priority queue."""
        self.searcher.print_pq()

    def plot(self, fig_type, **kwargs):
        """Plot training results.
        
        Args:
            fig_type (str): Type of figure to plot.
            **kwargs: Additional plotting arguments.
            
        Returns:
            Figure: Matplotlib figure.
        """
        return self.searcher.plot(fig_type, **kwargs)

    def predict(self):
        """Make predictions (not implemented)."""
        pass

class KD_DSCV_SPR(KD_DSCV):
    """KD_DSCV model with Physics-Informed Neural Networks.
    
    This class extends KD_DSCV with PINN capabilities for better
    equation discovery in sparse data settings.
    """

    def __init__(self, *args, config_out=None, **kwargs):
        """Initialize KD_DSCV_Pinn model.
        
        Args:
            *args: Positional arguments for KD_DSCV.
            config_out: Optional output configuration.
            **kwargs: Keyword arguments for KD_DSCV.
        """
        super().__init__(*args, config_out=None, **kwargs)
        self.base_config_file = "./discover/config/config_pde_pinn.json"
        self.set_config(config_out)
        self.is_realtime = False

    def set_config(self, config=None):
        """Set model configuration.

        Args:
            config: Configuration dictionary or path.
        """
        super().set_config(config)
        self.config_pinn = self.config["pinn"]
        self.config_task['task_type'] = 'pde_pinn'

    def import_dataset(
        self,
        dataset,
        *,
        sample=None,
        sample_ratio=0.1,
        colloc_num=None,
        random_state=42,
        noise_level=None,
        data_ratio=None,
        spline_sample=False,
        cut_quantile=None,
        dataset_name=None,
    ):
        """Import sparse/PINN data via :class:`~kd.dataset.PDEDataset`.

        Args:
            dataset: A :class:`kd.dataset.PDEDataset` instance.
            sample: Absolute number of training samples (takes precedence
                over ``sample_ratio`` when provided).
            sample_ratio: Fraction of available points to sample in ``(0, 1]``,
                default ``0.1``.
            colloc_num: Number of collocation points for PINN training; if
                ``None``, the default configured value is used.
            random_state: Random seed for reproducible sampling.
        """

        from kd.dataset import PDEDataset
        from .discover.adapter import DSCVSparseAdapter

        if not isinstance(dataset, PDEDataset):
            raise TypeError("dataset must be a PDEDataset instance")

        adapter = DSCVSparseAdapter(
            dataset,
            sample=sample,
            sample_ratio=sample_ratio,
            colloc_num=colloc_num,
            random_state=random_state,
            noise_level=noise_level or 0.0,
            data_ratio=data_ratio,
            spline_sample=spline_sample,
            cut_quantile=cut_quantile,
        )
        self.data_class = adapter
        self.dataset_ = dataset

        resolved_name = dataset_name or getattr(dataset, 'legacy_name', None) or \
            getattr(dataset, 'registry_name', None) or getattr(dataset, 'equation_name', None) or "custom_dataset"
        self.dataset = resolved_name
        if self.out_path is not None:
            self.out_path = os.path.join(
                self.out_path,
                f"discover_{resolved_name}_{self.seed}.csv"
            )

        self.setup()
        return self

    def make_pinn_model(self):
        """Create PINN model.
        
        Returns:
            PINN_model: PINN model object.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        model = PINN_model(
            self.out_path,
            self.config_pinn,
            self.config_task['dataset'],
            device
        )
        return model
    
    def setup(self):
        """Setup model components including PINN."""
        super().setup()
        self.denoise_pinn = self.make_pinn_model()
        self.denoise_pinn.import_outter_data(self.data_class.get_data())   
        
    def reset_up(self, clear_cache=True, reset_controller=True, new_seed=None):
        """Reset model components.
        
        Args:
            clear_cache (bool): Whether to clear program cache.
            reset_controller (bool): Whether to reset controller.
            new_seed (int): New random seed.
        """
        if clear_cache:
            Program.clear_cache()

        if new_seed is not None:
            self.set_seeds(new_seed)

        if reset_controller:
            self.controller = self.make_controller()
        
    def pretrain(self):
        """Pretrain PINN model with observations."""
        print("NN evaluator training with data")
        self.denoise_pinn.pretrain()
        Program.reset_task(self.denoise_pinn, self.config_pinn['generation_type'])
        
    def callIterPINN(self, n_epochs, verbose=True):
        """Run iterative PINN and PDE discovery.
        
        Args:
            n_epochs (int): Number of epochs.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Training results.
        """
        self.pretrain()

        last = False    
        last_best_p = None
        best_tokens = []
        prefix, _ = os.path.splitext(self.out_path)
        iter_num = self.config_pinn["iter_num"]
        eq_num = self.config_task.get('eq_num', 1)
        
        for i in range(iter_num):
            if i > 0:
                self.reset_up(reset_controller=False)
                bsz = self.config_training['batch_size']
                self.config_training['n_samples'] = 10*bsz

            print(f"The No.{i} pde discovery process")   
            results = [self.searcher.search(n_epochs=n_epochs, verbose=verbose, keep_history=False)]
            self.out_path = f"{prefix}_{i+1}.csv"
            best_p = [results[j]['program'] for j in range(len(results))]

            if len(best_tokens) > 0:
                new_best_p = []
                last_best_p = [from_tokens(best_tokens[t]) for t in range(len(results))]
                for j in range(len(results)):
                    if last_best_p[j].r_ridge > best_p[j].r_ridge:
                        new_best_p.append(last_best_p[j])
                    else:
                        new_best_p.append(best_p[j])
                best_p = new_best_p
               
            if i + 1 == iter_num:
                last = True
            
            self.pinn_train(best_p, count=i+1, coef=self.config_pinn['coef_pde'],
                            local_sample=self.config_pinn['local_sample'],
                            last=last)
            
            best_tokens = [best_p[j].tokens for j in range(len(results))]
            self.best_p = best_p

        print(f"The No.{iter_num} pde discovery process")
        self.reset_up(reset_controller=False)    
        return self.searcher.search(n_epochs=n_epochs, verbose=verbose, keep_history=False)
    
    def pinn_train(self, best_p, count, coef=0.1, local_sample=False, last=False):
        """Train PINN with discovered equation constraint.
        
        Args:
            best_p: Best program found.
            count (int): Iteration count.
            coef (float): Coefficient for PDE constraint.
            local_sample (bool): Whether to use local sampling.
            last (bool): Whether this is the last iteration.
        """
        self.denoise_pinn.train_pinn(best_p, count, coef=coef, local_sample=local_sample, last=last)
        Program.reset_task(self.denoise_pinn, self.config_pinn['generation_type'])
                
    def train(self, n_epochs, verbose):
        """Train model.
        
        Args:
            n_epochs (int): Number of epochs.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Training results.
        """
        self.setup()
        eq_num = self.config_task['eq_num'] 
        self.best_p = [None for _ in range(eq_num)]
        result = self.callIterPINN(n_epochs, verbose)
        return result

    def make_outter_data(self, x, y, domains, data_type='Sparse'):
        """Create data handler for external data.
        
        Args:
            x: Input features.
            y: Target values.
            domains: Domain boundaries.
            data_type (str): Type of data.
        """
        self.data_class = SparseData(x, y)
        self.data_class.process_data(domains)    

    def fit(self, x, y, domains, n_epochs=20, verbose=True, data_type='Sparse'):
        """Fit model to data.
        
        Args:
            x: Input features.
            y: Target values.
            domains: Domain boundaries.
            n_epochs (int): Number of epochs.
            verbose (bool): Whether to print progress.
            data_type (str): Type of data.
            
        Returns:
            dict: Training results.
        """
        self.make_outter_data(x, y, domains, data_type)
        return self.train(n_epochs=n_epochs, verbose=verbose)

    def predict(self, x):
        """Make predictions (not implemented)."""
        pass

    def fit_dataset(
        self,
        dataset,
        *,
        n_epochs: int = 20,
        verbose: bool = True,
        sample=None,
        sample_ratio: float = 0.1,
        colloc_num=None,
        random_state: int = 42,
        noise_level=None,
        data_ratio=None,
        spline_sample: bool = False,
        cut_quantile=None,
        dataset_name=None,
    ):
        """Convenience wrapper for Sparse+PINN training from :class:`PDEDataset`.

        This mirrors the style of the local PDE ``fit_dataset`` entry point
        while targeting the sparse/PINN DISCOVER task.
        """

        self.import_dataset(
            dataset,
            sample=sample,
            sample_ratio=sample_ratio,
            colloc_num=colloc_num,
            random_state=random_state,
            noise_level=noise_level,
            data_ratio=data_ratio,
            spline_sample=spline_sample,
            cut_quantile=cut_quantile,
            dataset_name=dataset_name,
        )
        return self.train(n_epochs=n_epochs, verbose=verbose)
