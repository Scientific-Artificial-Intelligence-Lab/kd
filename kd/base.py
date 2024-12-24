import inspect
from collections import defaultdict
import warnings
import numpy as np

class BaseEstimator:

    def __init__(self, **params):
        # Set parameters directly from keyword arguments
        self.set_params(**params)
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters for the framework and contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
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
        """
        Set parameters for this framework.

        Parameters
        ----------
        **params : dict
            Parameters mapped to values.

        Returns
        -------
        self : object
            Returns self.
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
        """
        Get parameter names for the framework.
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
        """
        String representation for debugging.
        """
        return f"{self.__class__.__name__}({self.get_params(deep=False)})"
