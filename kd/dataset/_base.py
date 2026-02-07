"""
Base IO code for all datasets
"""

import csv
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings
from typing import Any, Dict, List, Union, Optional, Tuple
from pathlib import Path
from importlib import resources
from abc import ABC, abstractmethod
from ._info import DatasetInfo
from scipy.interpolate import interp2d


DATA_MODULE = "kd.dataset.data"


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
    data_path = Path(file_path)
    # Attempt to load the file as a standard .mat (MATLAB 5) file using scipy.io
    try:
        # Try loading with scipy (for MATLAB version 5 and below)
        mat_data = sio.loadmat(data_path)
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

    def load_data(self, equation_name: str = None, file: str = None) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Loads PDE-related data from different file formats (CSV, MAT, NPY, NPZ).

        :param equation_name: The equation name (file name prefix) if file path is not provided.
        :param file: The full file path to load.
        :return: The loaded data as a NumPy array or a dictionary.
        :raises FileNotFoundError: If no matching data file is found.
        :raises ValueError: If the file format is unsupported.
        """
        if file:
            file_path = Path(file)
            if not file_path.exists():
                raise FileNotFoundError(f"Specified file does not exist: {file}")
        elif equation_name:
            file_path = self._find_file(equation_name)
            if file_path is None:
                raise FileNotFoundError(f"No data file found for equation: {equation_name}")
        else:
            raise ValueError("Either 'equation_name' or 'file' must be provided.")

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
    and contain necessary attributes, ensuring a consistent interface.
    """
    required_methods = {'get_data'}
    required_attributes = ('x', 't', 'usol')
    
    def __new__(cls, name, bases, dct):
        """
        Overrides class creation to enforce method and attribute requirements.
        """
                
        if not isinstance(cls.required_attributes, (set, list, tuple)):
            raise TypeError(f"{name}.required_attributes must be a set, list, or tuple.")
        
        # Ensure required methods are implemented
        for method in cls.required_methods:
            if method not in dct:
                raise TypeError(f'{name} must implement the method: {method}')
        
        # Ensure required attributes exist
        for attr in cls.required_attributes:
            if attr not in dct and not any(attr in base.__dict__ for base in bases):
                raise TypeError(f'{name} must contain the attribute: "{attr}"')
        
        return super().__new__(cls, name, bases, dct)


class MetaData(metaclass=MetaBase):
    """Base class to store metadata of a Partial Differential Equation (PDE) dataset."""
    
    x = None
    t = None
    usol = None
    
    def __init__(self, info: Any):
        """
        Initialize the metadata for a PDE dataset.

        :param info: Description of the equation.
        """
        self.info = info

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to attributes.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Attribute '{key}' not found in {self.__class__.__name__}.")

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Retrieve all PDE dataset information.
        """
        pass


class PDEDataset(MetaData):
    """ 
    A class representing a Partial Differential Equation (PDE) dataset, providing data access 
    and analysis functionality.
    """
    
    def __init__(
        self,
        equation_name: str,
        pde_data: Optional[Dict[str, Any]] = None,
        domain: Optional[Dict[str, Tuple[float, float]]] = None,
        epi: float = 0.0,
        x: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        usol: Optional[np.ndarray] = None,
        descr: Optional[DatasetInfo] = None,
        # N-D parameters
        fields_data: Optional[Dict[str, np.ndarray]] = None,
        coords_1d: Optional[Dict[str, np.ndarray]] = None,
        axis_order: Optional[List[str]] = None,
        target_field: str = "u",
        lhs_axis: str = "t",
    ):
        """
        Initializes the PDE dataset, supporting three input methods:
        1. Providing data through the `pde_data` dictionary (legacy 2D).
        2. Directly passing `x`, `t`, and `usol` arrays (legacy 2D).
        3. Using `fields_data` + `coords_1d` for N-D structured grids.

        :param equation_name: Name of the PDE.
        :param pde_data: Optional dictionary containing 'x', 't', and 'usol' data.
        :param domain: Dictionary defining the domain {variable: (min_value, max_value)}.
        :param epi: Additional parameter.
        :param x: Optional, spatial coordinate array (legacy 2D).
        :param t: Optional, temporal coordinate array (legacy 2D).
        :param usol: Optional, solution array u(x, t) (legacy 2D).
        :param descr: Metadata containing information about the PDE.
        :param fields_data: Dict mapping field names to N-D arrays (N-D mode).
        :param coords_1d: Dict mapping axis names to 1D coordinate arrays (N-D mode).
        :param axis_order: List of axis names defining array dimension order (N-D mode).
        :param target_field: Name of the target field in fields_data (default: "u").
        :param lhs_axis: Axis for LHS derivative (default: "t").
        """
        super().__init__(equation_name)

        # Determine construction mode
        has_nd_data = fields_data is not None or coords_1d is not None
        has_legacy_data = pde_data is not None or (x is not None and t is not None and usol is not None)

        if has_nd_data:
            # N-D mode
            self._init_nd_mode(
                fields_data=fields_data,
                coords_1d=coords_1d,
                axis_order=axis_order,
                target_field=target_field,
                lhs_axis=lhs_axis,
            )
        elif has_legacy_data:
            # Legacy 2D mode
            self._init_legacy_mode(pde_data=pde_data, x=x, t=t, usol=usol)
            # N-D attributes not set in legacy mode
            self.fields_data = None
            self.coords_1d = None
            self.axis_order = None
            self.target_field = None
            self.lhs_axis = None
        else:
            raise ValueError(
                "Either `pde_data`, `x/t/usol`, or `fields_data/coords_1d` must be provided."
            )

        self.u = self.usol
        self.equation_name = equation_name
        self.domain = domain
        self.epi = epi

    def _init_legacy_mode(
        self,
        pde_data: Optional[Dict[str, Any]],
        x: Optional[np.ndarray],
        t: Optional[np.ndarray],
        usol: Optional[np.ndarray],
    ) -> None:
        """Initialize in legacy 2D mode."""
        if pde_data is not None:
            self.x = np.asarray(pde_data.get("x"), dtype=float).flatten()
            self.t = np.asarray(pde_data.get("t"), dtype=float).flatten()
            self.usol = np.real(np.asarray(pde_data.get("usol")))
        else:
            self.x = np.asarray(x, dtype=float).flatten()
            self.t = np.asarray(t, dtype=float).flatten()
            self.usol = np.real(np.asarray(usol))

        # Validate shape
        if self.usol.shape != (len(self.x), len(self.t)):
            raise ValueError(
                f"usol dimensions {self.usol.shape} do not match x {len(self.x)} and t {len(self.t)}"
            )

    def _init_nd_mode(
        self,
        fields_data: Optional[Dict[str, np.ndarray]],
        coords_1d: Optional[Dict[str, np.ndarray]],
        axis_order: Optional[List[str]],
        target_field: str,
        lhs_axis: str,
    ) -> None:
        """Initialize in N-D mode with validation."""
        # Validate required parameters
        if fields_data is None or coords_1d is None:
            raise ValueError(
                "N-D mode requires both fields_data and coords_1d."
            )
        if axis_order is None:
            raise ValueError("axis_order is required for N-D mode.")

        # Convert to numpy arrays
        fields_data = {k: np.asarray(v) for k, v in fields_data.items()}
        coords_1d = {k: np.asarray(v).flatten() for k, v in coords_1d.items()}
        axis_order = list(axis_order)

        # Validate fields_data is not empty
        if len(fields_data) == 0:
            raise ValueError("fields_data cannot be empty.")

        # Validate axis_order is not empty
        if len(axis_order) == 0:
            raise ValueError("axis_order cannot be empty.")

        # Validate axis_order has no duplicates
        if len(axis_order) != len(set(axis_order)):
            duplicates = [a for a in axis_order if axis_order.count(a) > 1]
            raise ValueError(f"axis_order contains duplicate axes: {set(duplicates)}")

        # Validate axis_order matches coords_1d keys
        axis_set = set(axis_order)
        coords_set = set(coords_1d.keys())
        if axis_set != coords_set:
            missing = coords_set - axis_set
            extra = axis_set - coords_set
            msg = "axis_order and coords_1d keys must match."
            if missing:
                msg += f" Missing in axis_order: {missing}."
            if extra:
                msg += f" Extra in axis_order (not in coords_1d): {extra}."
            raise ValueError(msg)

        # Validate target_field exists
        if target_field not in fields_data:
            raise ValueError(
                f"target_field '{target_field}' not found in fields_data. "
                f"Available fields: {list(fields_data.keys())}"
            )

        # Validate lhs_axis exists (must be in axis_order, which equals coords_1d keys)
        if lhs_axis not in axis_order:
            raise ValueError(
                f"lhs_axis '{lhs_axis}' must be in axis_order {axis_order}."
            )

        # Compute expected shape from axis_order
        expected_shape = tuple(len(coords_1d[axis]) for axis in axis_order)

        # Validate all fields have consistent shape
        field_shapes = {name: arr.shape for name, arr in fields_data.items()}
        unique_shapes = set(field_shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(
                f"All fields must have the same shape. Got inconsistent shapes: {field_shapes}"
            )

        # Validate shape matches coords
        actual_shape = list(unique_shapes)[0] if unique_shapes else ()
        if actual_shape != expected_shape:
            raise ValueError(
                f"Field shape {actual_shape} does not match expected shape {expected_shape} "
                f"from coords_1d and axis_order."
            )

        # Store N-D attributes
        self.fields_data = fields_data
        self.coords_1d = coords_1d
        self.axis_order = axis_order
        self.target_field = target_field
        self.lhs_axis = lhs_axis

        # Set compatibility layer attributes
        self.x = coords_1d.get("x")
        self.t = coords_1d.get("t")
        self.usol = fields_data[target_field]

    def get_datapoint(self, x_id: int, t_id: int) -> Tuple[float, float, float]:
        """
        Retrieves the (x, t) coordinates and the corresponding solution value.

        :param x_id: Index in the spatial dimension.
        :param t_id: Index in the temporal dimension.
        :return: A tuple (x, t, usol) representing the coordinates and solution.
        """
        if not (0 <= x_id < len(self.x)) or not (0 <= t_id < len(self.t)):
            raise IndexError("Index out of range.")
        return self.x[x_id], self.t[t_id], self.usol[x_id, t_id]

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the dataset information as a dictionary.
        """
        return {'x': self.x, 't': self.t, 'usol': self.usol}
        
    def sample(self, n_samples: Union[int, float], method: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a subset of the data points using different sampling methods.

        :param n_samples: Number of samples to draw. If < 1, treated as a ratio.
        :param method: Sampling method to use ('random', 'uniform', 'spline').
        :return: A tuple (sampled_points, sampled_usol), where:
                - sampled_points is an array of shape (n, 2) with (x, t) pairs
                - sampled_usol is an array of shape (n,) with corresponding solution values
        """
        total_points = len(self.x) * len(self.t)
        
        # Convert ratio to absolute count if n_samples is a float < 1
        if isinstance(n_samples, float) and 0 < n_samples < 1:
            n_samples = int(total_points * n_samples)

        if n_samples > total_points:
            raise ValueError(f"Requested {n_samples} samples, but only {total_points} points available.")

        if method == 'random':
            indices = np.random.choice(total_points, n_samples, replace=False)
            x_samples, t_samples = np.unravel_index(indices, self.usol.shape)
            sampled_points = np.column_stack((self.x[x_samples], self.t[t_samples]))
            sampled_usol = self.usol[x_samples, t_samples]

        elif method == 'uniform':
            grid_size = int(np.sqrt(n_samples))
            x_indices = np.linspace(0, len(self.x) - 1, grid_size, dtype=int)
            t_indices = np.linspace(0, len(self.t) - 1, grid_size, dtype=int)
            x_samples, t_samples = np.meshgrid(x_indices, t_indices)
            sampled_points = np.column_stack((
                self.x[x_samples.flatten()],
                self.t[t_samples.flatten()]
            ))
            sampled_usol = self.usol[x_samples.flatten(), t_samples.flatten()]

        elif method == 'spline':
            grid_size = int(np.sqrt(n_samples))
            x_new = np.linspace(self.x.min(), self.x.max(), grid_size)
            t_new = np.linspace(self.t.min(), self.t.max(), grid_size)
            spline = interp2d(self.t, self.x, self.usol, kind='cubic')
            usol_new = spline(t_new, x_new)
            x_samples, t_samples = np.meshgrid(x_new, t_new)
            sampled_points = np.column_stack((x_samples.flatten(), t_samples.flatten()))
            sampled_usol = usol_new.flatten()

        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        return sampled_points, sampled_usol.reshape(-1, 1)
    
    def mesh(self, indexing='ij') -> np.ndarray:
        """
        Returns a 2D coordinate grid where each row is a combination of (x, t).
        Equivalent to np.meshgrid(x, t) but flattened into a 2D array.

        :return: A 2D array of shape (len(x) * len(t), 2) where each row is (x, t).
        """
        X, T = np.meshgrid(self.x, self.t, indexing=indexing)
        return np.column_stack([X.ravel(), T.ravel()])

    def mesh_bounds(self, indexing='ij') -> np.ndarray:
        X, T = np.meshgrid(self.x, self.t, indexing=indexing)
        mesh_data = np.column_stack([X.ravel(), T.ravel()])
        return mesh_data.min(0), mesh_data.max(0)
    
    def get_boundaries(self) -> Dict[str, Tuple[float, float]]:
        """Returns the minimum and maximum values for coordinates.

        In legacy 2D mode, returns {'x': (min, max), 't': (min, max)}.
        In N-D mode, returns boundaries for all coords_1d axes.
        """
        if self.coords_1d is not None:
            # N-D mode: return boundaries for all axes
            return {
                axis: (float(coords.min()), float(coords.max()))
                for axis, coords in self.coords_1d.items()
            }
        # Legacy 2D mode
        return {"x": (self.x.min(), self.x.max()), "t": (self.t.min(), self.t.max())}

    def get_domain(self) -> Dict[str, Tuple[float, float]]:
        """ Retrieves the domain definition. """
        return self.domain

    def get_derivative(self, axis: str = 'x') -> np.ndarray:
        """
        Computes the derivative of `usol` along the spatial or temporal axis.

        :param axis: Direction of differentiation ('x' or 't').
        :return: The computed derivative array.
        """
        if axis == 'x':
            return np.gradient(self.usol, axis=0)
        elif axis == 't':
            return np.gradient(self.usol, axis=1)
        else:
            raise ValueError("Invalid axis name, must be 'x' or 't'.")

    def get_range(self, x_range: Tuple[float, float], t_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """
        Retrieves subsets of x, t, and the corresponding solution within the specified range.

        :param x_range: Tuple representing the range of x values (min_x, max_x).
        :param t_range: Tuple representing the range of t values (min_t, max_t).
        :return: A dictionary containing subsets of x, t, and usol.
        """
        x_start, x_end = np.searchsorted(self.x, [x_range[0], x_range[1]])
        t_start, t_end = np.searchsorted(self.t, [t_range[0], t_range[1]])

        return {'x': self.x[x_start:x_end], 
                't': self.t[t_start:t_end], 
                'usol': self.usol[x_start:x_end, t_start:t_end]}

    def get_size(self) -> Tuple[int, int]:
        """ Returns the dimensions of the solution array (x dimension, t dimension). """
        return self.usol.shape

    def plot_solution(self) -> None:
        """
        Generates a heatmap visualization of the solution `usol` over (x, t) dimensions.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.usol, aspect='auto', origin='lower',
                   extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()])
        plt.colorbar(label='Solution u(x, t)')
        plt.xlabel('Time (t)')
        plt.ylabel('Space (x)')
        plt.title(f"Solution of {self.equation_name}")
        plt.show()

    def __repr__(self) -> str:
        """ Returns a string representation of the dataset. """
        return (f"PDEDataset(equation='{self.equation_name}', size={self.get_size()}, "
                f"boundaries={self.get_boundaries()})")
        
        
def load_burgers_equation():    
    warnings.warn(
        "load_burgers_equation() is deprecated; please use load_pde('burgers') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    descr = DatasetInfo(
        description = """
        Dataset for high-viscosity Burgers equation 
        ut=-uux+0.1uxx
        x∈[-8.0,8.0), t∈[0,10]
        nx=256, nt=201, u.shape=(256,201)
        Resource:DLGA-PDE: Discovery of PDEs with incomplete candidate library via combination of deep learning and genetic algorithm
        """
    )

    file_path = resources.files(DATA_MODULE) / "burgers2.mat"
    pde_data = load_mat_file(file_path)
    return PDEDataset(
        equation_name = 'burgers equation',
        descr = descr,
        pde_data = pde_data,
        domain = {'x': (-7.0, 7.0), 't': (1, 9)},
        epi = 1e-3
    )
    
def load_kdv_equation():
    warnings.warn(
        "load_kdv_equation() is deprecated; please use load_pde('kdv') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    descr = DatasetInfo(
        description = """
        Dataset for Korteweg-De Vries (KdV) equation with sin initial condition, actually a standardized form of Kdv_equation dataset
        ut=-uux-uxxx
        x∈[-20,20), t∈[0,40]
        nx=256, nt=201, u.shape=(256,201)
        Resource: PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Learning, pp20
        """
    )

    file_path = resources.files(DATA_MODULE) / "KdV_equation.mat"
    pde_data = load_mat_file(file_path)

    return PDEDataset(
        equation_name = 'kdv equation',
        descr = descr,
        pde_data = None,
        x = pde_data['x'],
        t = pde_data['tt'],
        usol = pde_data['uu'],
        domain = {'x': (-16, 16), 't': (5, 35)},
        epi = 1e-3
    )


def load_pde_dataset(
    filename: str,
    equation_name: str = 'PDE Dataset',
    x_key: str = 'x',
    t_key: str = 't',
    u_key: str = 'usol',
    domain: dict = None,
    epi: float = 1e-3,
    data_dir_module: str = "kd.dataset.data"
    ):
    """
    Convenience loader for PDE data stored in a MATLAB ``.mat`` file.

    This helper constructs the full file path, reads the ``.mat`` file into a
    dictionary, extracts the coordinate and solution arrays using the provided
    keys, normalises shapes, and wraps everything into a :class:`PDEDataset`.

    Args:
        filename: Name of the ``.mat`` file to load (e.g. ``\"my_data.mat\"``).
        equation_name: Human-readable name for this dataset.
        x_key: Key for the spatial coordinate in the ``.mat`` dictionary,
            default is ``\"x\"``.
        t_key: Key for the temporal coordinate, default is ``\"t\"``.
        u_key: Key for the solution field, default is ``\"usol\"``.
        domain: Optional analysis domain, e.g. ``{'x': (min_x, max_x), 't': (min_t, max_t)}``.
        epi: Recommended sparsity penalty for this dataset, default ``1e-3``.
        data_dir_module: Module path where data files are stored, default
            ``\"kd.dataset.data\"``.

    Returns:
        PDEDataset: A fully initialised dataset object that can be used by all
        KD models.
    """
    try:
        # 1. 自动构建文件的完整路径
        file_path = resources.files(data_dir_module) / filename
        
        # 2. 使用底层的加载器读取 .mat 文件
        pde_data = load_mat_file(file_path)

        # 3. 使用用户指定的键名，从加载的字典中提取数据
        x_data = np.asarray(pde_data[x_key], dtype=float).flatten()
        t_data = np.asarray(pde_data[t_key], dtype=float).flatten()
        u_data = np.asarray(pde_data[u_key])

        # 确保 u_data 的形状与 (len(x), len(t)) 对齐
        if u_data.shape == (len(t_data), len(x_data)):
            u_data = u_data.T

        # 4. 将所有信息送入 PDEDataset 进行标准化封装
        dataset = PDEDataset(
            equation_name=equation_name,
            pde_data=None, 
            x=x_data,
            t=t_data,
            usol=u_data,
            domain=domain, 
            epi=epi
        )
        print(f"成功加载数据集: {equation_name}，文件: {filename}")
        return dataset

    except FileNotFoundError:
        print(f"错误: 在默认数据目录中未找到文件 {filename}")
        return None
    except KeyError as e:
        print(f"错误: 文件 {filename} 中缺少必需的键: {e}。请检查您传入的 x_key, t_key, u_key 参数是否正确。")
        return None
    except Exception as e:
        print(f"错误: 加载或处理文件时发生未知错误: {e}")
        return None
