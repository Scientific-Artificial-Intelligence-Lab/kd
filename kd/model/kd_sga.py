"""
KD_SGA: SGA-PDE integration for the KD framework.

This module provides a unified interface for the SGA-PDE algorithm,
making it compatible with the KD framework's design patterns.
"""

import numpy as np
from ._base import BaseGa
from .sga.codes.sga import SGA
from .sga.codes import pde, tree, Data_generator, configure
from .sga.sga_refactored.data_utils import prepare_workspace
from .sga.sga_refactored.operators import get_default_library
from .sga.sga_refactored.workflow import inject_dependencies


class KD_SGA(BaseGa):
    """
    KD_SGA: SGA-PDE solver adapter for the KD framework.
    
    This class integrates the SGA-PDE algorithm into the KD framework,
    providing a unified interface that follows the framework's design patterns.
    It inherits from BaseGa and implements the required abstract methods.
    
    Parameters
    ----------
    num : int, default=20
        Population size for the SGA algorithm.
    depth : int, default=4
        Maximum depth of symbolic trees.
    width : int, default=5
        Maximum number of terms in PDE.
    p_var : float, default=0.5
        Probability of generating variables in trees.
    p_rep : float, default=1.0
        Replacement probability.
    p_mute : float, default=0.3
        Mutation probability.
    p_cro : float, default=0.5
        Crossover probability.
    data_mode : str, default='finite_difference'
        Mode for derivative calculation ('finite_difference' or 'autograd').
    
    Attributes
    ----------
    best_equation_ : object
        The best discovered equation object after fitting.
    best_aic_ : float
        The best AIC score achieved.
    eq_latex_ : str
        LaTeX representation of the discovered equation.
    """
    
    def __init__(self, 
                 num=20, 
                 depth=4, 
                 width=5, 
                 p_var=0.5, 
                 p_rep=1.0, 
                 p_mute=0.3, 
                 p_cro=0.5,
                 data_mode='finite_difference',
                 problem='Burgers'):
        """Initialize KD_SGA with given parameters."""
        self.num = num
        self.depth = depth
        self.width = width
        self.p_var = p_var
        self.p_rep = p_rep
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.data_mode = data_mode
        self.problem = problem
        
        # Initialize result attributes
        self.best_equation_ = None
        self.best_aic_ = None
        self.eq_latex_ = None

    def fit(self, X, y=None, u=None, t=None, x=None, max_gen=5, verbose=True):
        """
        Fit the SGA-PDE model to discover PDEs from data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data. Can be None if u, t, x are provided directly.
        y : array-like, shape (n_samples,), optional
            Target values. Not used in PDE discovery but kept for API compatibility.
        u : array-like, optional
            Solution field data. If None, uses default data from Data_generator.
        t : array-like, optional
            Time coordinates. If None, uses default data from Data_generator.
        x : array-like, optional
            Spatial coordinates. If None, uses default data from Data_generator.
        max_gen : int, default=5
            Maximum number of generations for SGA evolution.
        verbose : bool, default=True
            Whether to print detailed progress information.
            
        Returns
        -------
        self : KD_SGA
            Returns self for method chaining.
        """
        # === 修复配置文件副作用 / Fix Configuration Side Effects ===
        # 保存原始配置，设置新配置，训练后恢复
        # Save original config, set new config, restore after training
        original_problem = configure.problem
        
        # Handle data input - prioritize direct u,t,x input over X
        if u is None or t is None or x is None:
            if X is not None:
                # Try to extract u,t,x from X if it's a PDEDataset
                if hasattr(X, 'get_data'):
                    # X is a PDEDataset object
                    data_dict = X.get_data()
                    u, t, x = data_dict['usol'], data_dict['t'], data_dict['x']
                    if verbose:
                        print(f"[KD_SGA] Loaded data from PDEDataset: {X.equation_name}")
                    
                    # === 数据形状适配 / Data Shape Adaptation ===
                    # 统一数据加载可能返回与 SGA 内部期望不同的形状
                    # Unified data loading may return different shapes than SGA expects
                    if hasattr(X, 'equation_name') and 'chafee-infante' in X.equation_name:
                        # 对于 chafee-infante，需要确保数据形状与 SGA 内部一致
                        # For chafee-infante, ensure data shape matches SGA internal expectations
                        if u.shape != (256, 201):
                            if verbose:
                                print(f"[KD_SGA] Adapting data shape from {u.shape} to SGA expected format...")
                            
                            # 使用 SGA 内部的数据加载逻辑来获得正确的形状
                            # Use SGA internal data loading logic to get correct shape
                            configure.problem = 'chafee-infante'
                            Data_generator.ensure_data_generator_initialized()
                            u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
                            
                            if verbose:
                                print(f"[KD_SGA] Data shape adapted: u.shape={u.shape}")
                else:
                    # Fallback to original SGA data loading
                    if verbose:
                        print(f"[KD_SGA] Setting problem to '{self.problem}' and loading data...")
                    
                    # 临时设置configure.problem以加载正确的数据
                    # Temporarily set configure.problem to load correct data
                    configure.problem = self.problem
                    
                    # 确保Data_generator已初始化并加载正确的数据
                    # Ensure Data_generator is initialized and loads correct data
                    Data_generator.ensure_data_generator_initialized()
                    
                    u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
                    if verbose:
                        print(f"[KD_SGA] Loaded {self.problem} data since u,t,x not provided")
            else:
                if verbose:
                    print(f"[KD_SGA] Setting problem to '{self.problem}' and loading data...")
                
                # 确保Data_generator已初始化并加载正确的数据
                # Ensure Data_generator is initialized and loads correct data
                Data_generator.ensure_data_generator_initialized()
                
                u, t, x = Data_generator.u, Data_generator.t, Data_generator.x
                if verbose:
                    print(f"[KD_SGA] Using {self.problem} data")
        
        if verbose:
            print(f"[KD_SGA] Data loaded: u.shape={u.shape}")

        # 1. Data preprocessing and derivative calculation
        workspace = prepare_workspace(u, t, x, mode=self.data_mode)
        symbol_library = get_default_library(workspace)

        # 2. Dependency injection
        inject_dependencies(workspace, symbol_library)
        if verbose:
            print("[KD_SGA] Dependencies injected.")

        # 3. Instantiate and run SGA
        sga_instance = SGA(
            num=self.num,
            depth=self.depth,
            width=self.width,
            p_var=self.p_var,
            p_rep=self.p_rep,
            p_mute=self.p_mute,
            p_cro=self.p_cro
        )
        sga_instance.run(gen=max_gen)
        best_eq_obj, best_aic = sga_instance.the_best()
        
        # Store results
        self.best_equation_ = best_eq_obj
        self.best_aic_ = best_aic
        
        if verbose:
            print("[KD_SGA] Best AIC score:", best_aic)
            print("[KD_SGA] Discovered Equation:", best_eq_obj.concise_visualize())
        
        # Generate LaTeX representation
        try:
            self.eq_latex_ = self._generate_latex()
        except Exception as e:
            if verbose:
                print(f"[KD_SGA] Warning: LaTeX generation failed: {e}")
            self.eq_latex_ = None
        
        # === 恢复原始配置 / Restore Original Configuration ===
        # 恢复configure.problem以避免影响其他代码
        # Restore configure.problem to avoid affecting other code
        try:
            configure.problem = original_problem
            if verbose:
                print(f"[KD_SGA] Configuration restored to original problem: {original_problem}")
        except Exception as e:
            if verbose:
                print(f"[KD_SGA] Warning: Failed to restore configuration: {e}")
            
        return self

    def predict(self, X):
        """
        Make predictions using the discovered equation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for prediction.
            
        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted values.
            
        Note
        ----
        This is a placeholder implementation. The actual prediction
        would depend on how the discovered PDE should be evaluated.
        """
        if self.best_equation_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Placeholder implementation
        # In practice, this would evaluate the discovered PDE on new data
        return np.zeros(X.shape[0])

    def score(self, X, y=None):
        """
        Return the AIC score of the discovered equation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        score : float
            The negative AIC score (higher is better for sklearn compatibility).
        """
        if self.best_aic_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Return negative AIC for sklearn compatibility (higher is better)
        return -self.best_aic_

    def get_equation_latex(self):
        """
        Get the LaTeX representation of the discovered equation.
        
        Returns
        -------
        latex_str : str or None
            LaTeX string of the equation, or None if not available.
        """
        return self.eq_latex_

    def get_equation_string(self):
        """
        Get the string representation of the discovered equation.
        
        Returns
        -------
        eq_str : str or None
            String representation of the equation, or None if not fitted.
        """
        if self.best_equation_ is None:
            return None
        return self.best_equation_.concise_visualize()

    def _generate_latex(self):
        """
        Generate LaTeX representation of the discovered equation.
        
        Uses the integrated SGA equation renderer from kd.viz.
        
        Returns
        -------
        latex_str : str
            LaTeX representation of the equation.
        """
        if self.best_equation_ is None:
            return None
            
        try:
            from kd.viz.sga_eq2latex import sga_eq2latex
            return sga_eq2latex(self.best_equation_, lhs_name="u_t")
        except ImportError:
            # Fallback to simple string formatting
            eq_str = self.best_equation_.concise_visualize()
            return f"$u_t = {eq_str}$"

    def __repr__(self):
        """Return string representation of the estimator."""
        return f"KD_SGA(num={self.num}, depth={self.depth}, width={self.width})"
