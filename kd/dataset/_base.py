"""
Base IO code for all datasets
"""

import csv
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Any, Dict, Union, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from ..utils import Attrdict
from _info import DatasetInfo
from scipy.interpolate import interp2d


DATA_MODULE = "kd.datasets.data"


def _convert_data_dataframe(
    data, target, feature_names, target_names, sparse_data=False
):
    # If the data is not sparse, create a regular DataFrame for features.
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    else:
        # If the data is sparse, create a sparse DataFrame for features.
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    # Create a DataFrame for the target variable with appropriate column names.
    target_df = pd.DataFrame(target, columns=target_names)
    
    # Concatenate the data and target DataFrames along columns (axis=1) to create a combined DataFrame.
    combined_df = pd.concat([data_df, target_df], axis=1)
    
    # Separate the feature columns (X) and the target columns (y) from the combined DataFrame.
    X = combined_df[feature_names]
    y = combined_df[target_names]
    
    # If there is only one target variable (i.e., y has only one column), simplify y to a 1D Series.
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    
    # Return the combined DataFrame, features (X), and target (y).
    return combined_df, X, y


def load_csv_data(data_file_path: str, encoding: str = "utf-8", has_header: bool = True) -> np.ndarray:
    """
    Reads a CSV file and returns the data as a NumPy array, automatically determining whether there is a header.

    Depending on the value of `has_header`, the function either skips the first row (if it is a header) 
    or includes it as data.

    Args:
        data_file_path (str): The path to the CSV file.
        encoding (str): The file encoding, default is 'utf-8'.
        has_header (bool): Whether the CSV file contains a header row, default is True.

    Returns:
        np.ndarray: A NumPy array containing the data. If `has_header` is True, the first row will be excluded.

    Example:
        >>> data = load_csv_data('example.csv')
        >>> print(data)
        [[25. 85.]
         [22. 90.]
         [23. 88.]]
    """
    # Create a Path object for the file path
    data_path = Path(data_file_path)

    # Open the CSV file and read the data
    with data_path.open("r", encoding=encoding) as f:
        data = csv.reader(f)
        
        # Read all rows from the CSV
        rows = list(data)
        
        if has_header:
            # If the file has a header, remove the first row
            header = rows[0]  # Store header if needed (can be returned or processed)
            data_rows = rows[1:]
        else:
            # If no header, use all rows as data
            data_rows = rows
        
        # Convert the data into a NumPy array
        data_array = np.array(data_rows, dtype=np.float32)  # Assuming the data is numeric; handle further conversions if needed
        
    return data_array


def load_mat_file(file_path: str) -> Dict[str, Any]:
    """
    Parses a .mat file (MATLAB format) and returns its content as a Python dictionary.
    Supports both older .mat files (MATLAB 5) and newer ones (MATLAB 7.3 or HDF5 format).
    
    Args:
        file_path (str): The path to the .mat file to be loaded.

    Returns:
        Dict[str, Any]: A dictionary where keys are variable names and values are corresponding data arrays.
    
    Raises:
        ValueError: If the file format is not supported or if there's an error in reading the file.
    """
    # Attempt to load the file as a standard .mat (MATLAB 5) file using scipy.io
    try:
        # Try loading with scipy (for MATLAB version 5 and below)
        mat_data = sio.loadmat(file_path)
        # Remove MATLAB-specific metadata (keys like __header__, __version__, __globals__)
        mat_data_clean = {key: value for key, value in mat_data.items() if not key.startswith('__')}
        return mat_data_clean
    except NotImplementedError:
        # This error will occur if scipy cannot handle the file (e.g., MATLAB version > 7.3)
        raise ValueError("The .mat file is of an unsupported format (likely version 7.3 or higher).")

  
def load_numpy_data(file_path: str) -> Union[np.ndarray, dict]:
    """
    Loads NumPy data from a `.npy` or `.npz` file and returns it as a NumPy array or a dictionary (for `.npz` files).

    Args:
        file_path (str): The path to the `.npy` or `.npz` file.

    Returns:
        np.ndarray or dict: If the file is a `.npy` file, a NumPy array is returned. 
                             If the file is a `.npz` file, a dictionary of arrays is returned.

    Example:
        >>> data = load_numpy_data('data.npy')
        >>> print(data)
        [1. 2. 3. 4.]
        
        >>> data = load_numpy_data('data.npz')
        >>> print(data['arr_0'])
        [1. 2. 3. 4.]
    """
    if file_path.endswith('.npy'):
        # Load a single NumPy array from a .npy file
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        # Load a NumPy compressed archive (.npz) and return as a dictionary of arrays
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


class FiniteDifferenceOperator:
    def __init__(self, u, dx, dim):
        """
        Initializes the class with the grid values, grid spacing, and dimension.
        :param u: Input array representing the function values.
        :param dx: Grid spacing.
        :param dim: Spatial dimension.
        """
        self.u = u
        self.dx = dx
        self.dim = dim
        self.n, self.m = u.shape
        self.ux = np.zeros_like(u)

    def _finite_diff_1st(self):
        """
        Computes the first-order derivative.
        """
        self.ux[1:self.n-1, :] = (self.u[2:self.n, :] - self.u[0:self.n-2, :]) / (2 * self.dx)
        self.ux[0, :] = (-3.0 / 2 * self.u[0, :] + 2 * self.u[1, :] - 1/2 * self.u[2, :]) / self.dx
        self.ux[self.n-1, :] = (3.0 / 2 * self.u[self.n-1, :] - 2 * self.u[self.n-2, :] + 1/2 * self.u[self.n-3, :]) / self.dx

    def _finite_diff_2nd(self):
        """
        Computes the second-order derivative.
        """
        self.ux[1:self.n-1, :] = (self.u[2:self.n, :] - 2 * self.u[1:self.n-1, :] + self.u[0:self.n-2, :]) / self.dx ** 2
        self.ux[0, :] = (2 * self.u[0, :] - 5 * self.u[1, :] + 4 * self.u[2, :] - self.u[3, :]) / self.dx ** 2
        self.ux[self.n-1, :] = (2 * self.u[self.n-1, :] - 5 * self.u[self.n-2, :] + 4 * self.u[self.n-3, :] - self.u[self.n-4, :]) / self.dx ** 2

    def _finite_diff_3rd(self):
        """
        Computes the third-order derivative.
        """
        self._finite_diff_2nd()  # First compute the second-order derivative
        temp = FiniteDifferenceOperator(self.ux, self.dx, self.dim)
        temp._finite_diff_1st()  # Apply first-order difference to the second derivative
        self.ux = temp.ux

    def _finite_diff_4th(self):
        """
        Computes the fourth-order derivative.
        """
        self._finite_diff_2nd()  # First compute the second-order derivative
        temp = FiniteDifferenceOperator(self.ux, self.dx, self.dim)
        temp._finite_diff_2nd()  # Apply second-order difference to the second derivative
        self.ux = temp.ux

    def compute(self, order):
        """
        Computes the derivative of the specified order.
        :param order: Order of the derivative (1, 2, 3, 4)
        :return: The computed derivative.
        """
        if order == 1:
            self._finite_diff_1st()
        elif order == 2:
            self._finite_diff_2nd()
        elif order == 3:
            self._finite_diff_3rd()
        elif order == 4:
            self._finite_diff_4th()
        else:
            raise ValueError("Only orders 1, 2, 3, and 4 are supported.")
        return self.ux

    @staticmethod
    def diff(u, dx, order=1):
        """
        Static method for convenient direct derivative computation.
        :param u: Input array.
        :param dx: Grid spacing.
        :param order: Order of the derivative.
        :return: The computed derivative.
        """
        operator = FiniteDifferenceOperator(u, dx, dim=u.ndim)
        return operator.compute(order)


class BaseDataLoader(ABC):
    """
    Abstract base class defining the interface for data loaders.
    """
    @abstractmethod
    def load_data(self):
        pass


class PDEDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str):
        """
        Initializes the data loader.

        :param data_dir: Directory where data files are stored.
        """
        self.data_dir = Path(data_dir)

    def load_data(self, equation_name: str) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Loads PDE-related data from different file formats (CSV, MAT, NPY, NPZ).

        :param equation_name: The equation name (file name prefix).
        :return: The loaded data as a NumPy array or a dictionary.
        :raises FileNotFoundError: If no matching data file is found.
        :raises ValueError: If the file format is unsupported.
        """
        # Find the matching file based on the equation name
        file_path = self._find_file(equation_name)
        if file_path is None:
            raise FileNotFoundError(f"No data file found for equation: {equation_name}")

        # Load the file based on its extension
        if file_path.suffix == ".csv":
            return load_csv_data(str(file_path))
        elif file_path.suffix == ".mat":
            return load_mat_file(str(file_path))
        elif file_path.suffix in [".npy", ".npz"]:
            return load_numpy_data(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _find_file(self, equation_name: str) -> Union[Path, None]:
        """
        Searches for a file that matches the given equation name in the data directory.

        :param equation_name: The equation name (file name prefix).
        :return: The path to the matching file, or None if no file is found.
        """
        for ext in [".csv", ".mat", ".npy", ".npz"]:
            file_path = self.data_dir / f"{equation_name}{ext}"
            if file_path.exists():
                return file_path
        return None


class MetaBase(type):
    """ 
    Metaclass to enforce that subclasses implement required methods 
    and contain necessary attributes. This ensures a consistent interface 
    for all subclasses.
    """
    
    def __new__(cls, name, bases, dct):
        """
        Overrides the default method for creating new class definitions.
        
        This method checks that any subclass of `MetaBase` implements the required 
        methods and contains the necessary attributes. If any condition is not met, 
        an error is raised to enforce conformity.

        :param cls: The metaclass instance.
        :param name: Name of the new class being created.
        :param bases: Tuple containing base classes of the new class.
        :param dct: Dictionary containing attributes and methods of the new class.
        :return: The newly created class.
        """

        # List of required methods that every subclass must implement
        required_methods = ['get_datapoint', 'get_domain', 'get_size']

        # List of required attributes that every subclass must have
        required_attributes = ['x', 't', 'usol']

        # Ensure that the subclass implements all required methods
        for method in required_methods:
            if method not in dct:
                raise TypeError(f'{name} class must implement the method: {method}')

        # Ensure that the subclass contains all required attributes
        for attr in required_attributes:
            # Check if the attribute exists in the subclass or any of its base classes
            if not any(hasattr(base, attr) for base in bases) and attr not in dct:
                raise TypeError(f'{name} class must contain the attribute: "{attr}"')

        # Create and return the new class
        return super().__new__(cls, name, bases, dct)


class MetaData(metaclass=MetaBase):
    """ Base class to store metadata of a Partial Differential Equation (PDE) dataset """

    def __init__(self, equation_name: str, descr: DatasetInfo):
        """
        :param equation_name: Name of the PDE
        :param descr: Description of the equation
        """
        self.equation_name = equation_name
        self.descr = descr


class PDEDataset(MetaData):
    """ 
    Class representing a Partial Differential Equation (PDE) dataset, 
    providing access and analysis functionalities.
    """

    def __init__(self, equation_name: str, descr: DatasetInfo, pde_data: Dict[str, Any], 
                 domain: Tuple[Tuple[float, float], Tuple[float, float]], epi: float):
        """
        Initialize the PDE dataset with given parameters.

        :param equation_name: Name of the PDE.
        :param descr: Description of the equation.
        :param pde_data: Dictionary containing 'x', 't', and 'usol' data arrays.
                         - 'x': Array representing spatial coordinates.
                         - 't': Array representing time coordinates.
                         - 'usol': 2D array representing the solution u(x, t).
        :param domain: Tuple representing domain conditions as ((x_min, x_max), (t_min, t_max)).
        :param epi: Additional parameter related to the dataset.
        """
        super().__init__(equation_name, descr)
        
        # Read pde_data and validate its completeness
        try:
            self.x: np.ndarray = np.array(pde_data['x'])  # Spatial coordinates
            self.t: np.ndarray = np.array(pde_data['t'])  # Time coordinates
            self.usol: np.ndarray = np.array(pde_data['usol'])  # Solution array
        except KeyError as e:
            raise ValueError(f"Missing key in pde_data: {e}")

        # Assign additional parameters
        self.domain: Tuple[Tuple[float, float], Tuple[float, float]] = domain  # Domain range
        self.epi: float = epi  # Extra parameter

        # Ensure 'usol' dimensions match 'x' and 't'
        if self.usol.shape != (len(self.x), len(self.t)):
            raise ValueError("Dimensions of 'usol' must match 'x' and 't'")

    def get_datapoint(self, x_id: int, t_id: int) -> Tuple[float, float, float]:
        """
        Retrieve the (x, t) coordinates and the corresponding solution value at a given index.

        :param x_id: Index of the spatial coordinate.
        :param t_id: Index of the time coordinate.
        :return: Tuple (x, t, usol) containing the coordinate and solution value.
        """
        return self.x[x_id], self.t[t_id], self.usol[x_id, t_id]
    
    def get_boundaries(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the minimum and maximum boundaries of x and t.

        :return: Dictionary with min and max values for x and t.
        """
        return {'x': (self.x.min(), self.x.max()), 't': (self.t.min(), self.t.max())}

    def get_domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Retrieve the domain conditions.

        :return: Tuple containing the domain ranges ((x_min, x_max), (t_min, t_max)).
        """
        return self.domain

    def get_range(self, x_range: Tuple[float, float], t_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """ 
        Retrieve x and t data within the specified ranges along with the corresponding solution values.

        :param x_range: Tuple (min_x, max_x) specifying the x range.
        :param t_range: Tuple (min_t, max_t) specifying the t range.
        :return: Dictionary containing:
                 - 'x': Subset of x values within the given range.
                 - 't': Subset of t values within the given range.
                 - 'usol': Corresponding subset of the solution array.
        """
        x_mask = (self.x >= x_range[0]) & (self.x <= x_range[1])
        t_mask = (self.t >= t_range[0]) & (self.t <= t_range[1])

        sub_x = self.x[x_mask]
        sub_t = self.t[t_mask]
        sub_usol = self.usol[np.ix_(x_mask, t_mask)]

        return {'x': sub_x, 't': sub_t, 'usol': sub_usol}

    def get_size(self) -> Tuple[int, int]:
        """
        Retrieve the size of the solution array.

        :return: Tuple (number of spatial points, number of time points).
        """
        return self.usol.shape

    def get_data(self) -> Dict[str, Any]:
        """
        Retrieve all PDE dataset information.

        :return: Dictionary containing all dataset attributes:
                 - 'x', 't', 'usol': The core dataset arrays.
                 - 'x_low', 'x_up': The spatial domain boundaries.
                 - 't_low', 't_up': The time domain boundaries.
                 - 'epi': Additional dataset parameter.
        """
        (x_low, x_up), (t_low, t_up) = self.domain
        return {
            'x': self.x,
            't': self.t,
            'usol': self.usol,
            'x_low': x_low,
            'x_up': x_up,
            't_low': t_low,
            't_up': t_up,
            'epi': self.epi
        }

    def generate_grid(self, x_points: int = 100, t_points: int = 100) -> None:
        """
        Generate an evenly spaced grid for x and t if they are not provided.

        :param x_points: Number of spatial points.
        :param t_points: Number of time points.
        """
        (x_min, x_max), (t_min, t_max) = self.domain
        self.x = np.linspace(x_min, x_max, x_points)
        self.t = np.linspace(t_min, t_max, t_points)
        self.usol = np.zeros((x_points, t_points))  # Placeholder

    def extract_time_slice(self, t_value: float) -> np.ndarray:
        """
        Extract the solution values at a specific time.

        :param t_value: The time value to extract.
        :return: Array of solution values at the specified time.
        """
        t_index = np.abs(self.t - t_value).argmin()  # Find the closest time index
        return self.usol[:, t_index]
    
    def info(self) -> DatasetInfo:
        return self.descr
    
    def __repr__(self) -> str:
        """
        Return a formatted string representation of the dataset.

        :return: String summarizing the dataset properties.
        """
        return (f"PDEDataset(equation='{self.equation_name}', size={self.get_size()}, "
                f"boundaries={self.get_boundaries()})")