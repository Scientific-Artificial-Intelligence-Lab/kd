"""Base class for all estimators in KD.

This class provides basic functionality for all estimators in the KD framework,
including parameter management and string representation. It is inspired by
scikit-learn's BaseEstimator.

Attributes:
    None
"""
import inspect
from collections import defaultdict
import warnings
import numpy as np

class BaseEstimator:
    """Base class for all estimators in KD.
    
    This class should not be used directly. Instead, use derived classes.
    All estimators should inherit from this class.
    
    Attributes:
        None
    """

    def __init__(self, **params):
        """Initialize self.
        
        Args:
            **params: Arbitrary keyword arguments that will be set as parameters
                of the estimator.
        """
        self.set_params(**params)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Args:
            deep (bool, optional): If True, will return the parameters for this
                estimator and contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.

        Raises:
            ValueError: If trying to set an invalid parameter.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for {self}.")
            
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        
        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator.
        
        Returns:
            list: List of parameter names in the format ``[p1, p2, p3...]``.
            
        Note:
            Parameters are fetched directly from the class ``__init__`` method.
        """
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() 
                      if p.name != "self" and p.kind != p.VAR_KEYWORD]
        
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls} should not use *args in constructor; parameters must be specified."
                )
        
        return sorted([p.name for p in parameters])

    def __repr__(self):
        """Return a string representation of the estimator.
        
        Returns:
            str: String representation.
        """
        return f"{self.__class__.__name__}({self.get_params(deep=False)})"
