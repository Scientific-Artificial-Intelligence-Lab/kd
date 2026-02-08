"""Data handling module for PDE-related datasets.

This module provides classes for handling different types of PDE data:
    - chafee-infante: Chafee-Infante equation data (301x200)
    - Burgers: Burgers equation data
    - Kdv: Korteweg-de Vries equation data (512x201)

The module supports both regular grid data (RegularData) and sparse sampling data (SparseData).
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.io as scio
from pyDOE import lhs
from .utils.utils_fd import FiniteDiff

data_list = [
    'chafee-infante', 'Burgers', 'Kdv'
]

class BaseDataHandler(ABC):
    """Abstract base class for data handling.
    
    This class provides basic data loading and cleaning functionality.
    Subclasses should implement specific data processing logic.
    
    Attributes:
        source: Data source (file path or DataFrame)
        data: Loaded and processed data
    """

    def __init__(self, source=None):
        """Initialize data handler.
        
        Args:
            source: Data source, can be file path or DataFrame
        """
        self.source = source
        self.data = None

    def load_data(self):
        """Load data from source.
        
        Supports loading from:
            - CSV files
            - Pandas DataFrames
            
        Raises:
            ValueError: If source format is not supported
        """
        if isinstance(self.source, str):
            try:
                self.data = pd.read_csv(self.source)
                print("Data loaded successfully from file.")
            except Exception as e:
                print(f"Error loading data: {e}")
        elif isinstance(self.source, pd.DataFrame):
            self.data = self.source.copy()
            print("Data loaded from user-provided DataFrame.")
        else:
            raise ValueError("Unsupported data source format. Please provide a file path or DataFrame.")

    def clean_data(self):
        """Clean data by removing NaN values and duplicates."""
        if self.data is not None:
            self.data.dropna(inplace=True)
            self.data.drop_duplicates(inplace=True)
            print("Data cleaned.")
        else:
            print("No data to clean.")

    @abstractmethod
    def process_data(self):
        """Process data according to specific requirements.
        
        To be implemented by subclasses.
        """
        pass

    def get_data(self):
        """Get the processed data.
        
        Returns:
            dict: Processed data dictionary
        """
        return self.data

class RegularData(BaseDataHandler):
    """Handler for regular grid PDE data.
    
    This class handles data from PDE solutions on regular grids, including:
        - chafee-infante (301x200)
        - Burgers equation
        - KdV equation (512x201)
    
    Attributes:
        X (list): Spatial coordinates
        sym_true (str): Symbolic form of the true equation
        n_input_dim (int): Number of spatial dimensions (default: 1)
        u (ndarray): Solution values
        ut (ndarray): Time derivatives
    """
    
    def __init__(self, source):
        """Initialize regular data handler.
        
        Args:
            source (str): Name of the dataset to load
            
        Raises:
            AssertionError: If dataset name not in supported list
        """
        super().__init__(source)
        
        self.X = []
        self.sym_true = None
        self.n_input_dim = 1  # default 1 spatial dimension
        if isinstance(self.source, str):
            assert self.source in data_list, "Provided data is not included"
              
    def process_data(self):
        """Process regular grid data."""
        super().process_data()
        pass
    
    def load_data(self):
        """Load data from predefined datasets.
        
        The following datasets are supported:
            - chafee-infante: Chafee-Infante equation data
            - Burgers: Burgers equation data
            - Kdv: Korteweg-de Vries equation data
            
        Raises:
            ValueError: If dataset is not supported.
        """
        folder_name = './kd/data_file'
        if self.source == 'chafee-infante':  # 301*200
            u = np.load(f"{folder_name}/chafee_infante_CI.npy")
            x = np.load(f"{folder_name}/chafee_infante_x.npy").reshape(-1,1)
            t = np.load(f"{folder_name}/chafee_infante_t.npy").reshape(-1,1)
            self.sym_true = 'add,add,u1,n3,u1,diff2,u1,x1'

        elif self.source == 'Burgers':
            data = scio.loadmat(f"{folder_name}/burgers.mat")
            u = data.get("usol")
            x = np.squeeze(data.get("x")).reshape(-1,1)
            t = np.squeeze(data.get("t").reshape(-1,1))
            self.sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'

        elif self.source == 'Kdv':      
            data = scio.loadmat('./discover/task/pde/data_new/Kdv.mat')
            u = data.get("uu")
            n,m = u.shape  # 512x201
            x = np.squeeze(data.get("x")).reshape(-1,1)
            t = np.squeeze(data.get("tt").reshape(-1,1))
            dt = t[1]-t[0]
            dx = x[1]-x[0]
            self.sym_true = 'add,mul,u1,diff,u1,x1,diff3,u1,x1'
        else:
            raise NotImplementedError("Dataset not implemented")
            
        n, m = u.shape
        self.ut = np.zeros((n, m))
        dt = t[1]-t[0]
        self.X.append(x)
        self.u = u

        # Compute time derivatives
        for idx in range(n):
            self.ut[idx, :] = FiniteDiff(u[idx, :], dt)
        
        self.data = {
            'u': self.u,
            'ut': self.ut,
            'X': self.X,
            'n_input_dim': self.n_input_dim,
            'sym_true': self.sym_true
        }
        
    def info(self):
        """Print information about the loaded data."""
        print()

class SparseData(BaseDataHandler):
    """Handler for sparsely sampled PDE data.
    
    This class handles data from sparse sampling points, typically used
    when regular grid data is not available. It includes functionality
    for train/validation splitting and collocation point sampling.
    
    Attributes:
        xt (ndarray): Space-time coordinates
        x (ndarray): Spatial coordinates
        t (ndarray): Time coordinates
        u (ndarray): Solution values (shape: (N, 1))
        colloc_num (int): Number of collocation points (default: 20000)
    """
    
    def __init__(self, xt, u, *args, **kwargs):
        """Initialize sparse data handler.
        
        Args:
            xt (ndarray): Space-time coordinates
            u (ndarray): Solution values
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)    
        
        self.xt = xt
        self.x = xt[:,:-1]  # spatial coordinates
        self.t = xt[:,-1:]  # time coordinates
        self.u = u  # shape = (N, 1)
        
        self.colloc_num = 20000
        
    def process_data(self, domains):
        """Process sparse data.
        
        Args:
            domains (tuple): (lower_bound, upper_bound) for the domain
            
        The processing includes:
            1. Train/validation split (80/20)
            2. Latin Hypercube Sampling for collocation points
        """
        super().process_data()
        
        # 1. train test split
        Split_TrainVal = 0.8
        Num_d = self.u.shape[0]
        N_u_train = int(Num_d*Split_TrainVal)
        idx_train = np.random.choice(Num_d, N_u_train, replace=False)
        self.X_u_train = self.xt[idx_train,:]
        self.u_train = self.u[idx_train,:]
        idx_val = np.setdiff1d(np.arange(Num_d), idx_train, assume_unique=True)
        self.X_u_val = self.xt[idx_val,:]
        self.u_val = self.u[idx_val,:]
        
        # 2. Collocation points sampling
        lb, ub = domains[0], domains[1]
        n_dims = self.xt.shape[-1]
        X_f_train = lb + (ub-lb)*lhs(n_dims, self.colloc_num)
        self.X_f_train = np.vstack((X_f_train, self.X_u_train))   

        self.data = {
            'X_u_train': self.X_u_train,
            'u_train': self.u_train,
            'X_f_train': self.X_f_train,
            'X_u_val': self.X_u_val,
            'u_val': self.u_val,
            'lb': lb,
            'ub': ub
        }
        
    

