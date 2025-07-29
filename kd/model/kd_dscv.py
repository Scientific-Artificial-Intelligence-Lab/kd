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
        # Todo: Implement other data types

    def fit(self, X, y, domains=[], data_type='Sparse'):
        """Fit model to data.
        
        Args:
            X: Input features.
            y: Target values.
            domains (list): Domain boundaries.
            data_type (str): Type of data.
        """
        self.make_outter_data(X, y, domains, data_type)
        self.setup()

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
        if self.config_gp_agg.pop("run_gp_agg", False):
            from discover.aggregator import gpAggregator
            gp_aggregator = gpAggregator(self.prior,
                                         self.pool,
                                         self.config_gp_agg)
        else:
            gp_aggregator = None
        return gp_aggregator
    
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
        
        assert self.data_class is not None, "Dataset should be made before setting task"
        set_task(self.config_task, self.data_class.get_data())

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
