"""
Base IO code for all datasets
"""

import csv
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Any, Dict, Union, Optional
from pathlib import Path


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

  
def load_numpy_data():
    pass


def load_burgers():
    pass


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
