class AttrDict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    Reference: https://stackoverflow.com/a/14620633

    Example:
        >>> data = AttrDict(a=1, b=2)  # Create an AttrDict object
        >>> print(data.a)  # Access dictionary key 'a' as an attribute
        1
        >>> data.c = 3  # Add a new key-value pair
        >>> print(data['c'])  # Access dictionary key 'c' using dictionary syntax
        3
        >>> del data.a  # Delete a key-value pair
        >>> print(data.a)  # Raises AttributeError for non-existent key
        AttributeError: 'AttrDict' object has no attribute 'a'
        >>> print(data.keys())  # Use dictionary methods like keys()
        dict_keys(['b', 'c'])
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the dictionary and bind its internal dictionary to the instance.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        # Avoid shadowing by explicitly setting an internal reference for attributes
        object.__setattr__(self, '_internal_dict', self)

    def __getattr__(self, item):
        """
        Provide attribute-style access for dictionary keys.
        Raise AttributeError if the key is not present.
        """
        try:
            return self[item]  # Attempt to retrieve the item using key access
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Allow setting attributes, which translates to setting dictionary keys.
        """
        self[key] = value

    def __delattr__(self, item):
        """
        Allow deletion of attributes, which translates to deleting dictionary keys.
        Raise AttributeError if the key does not exist.
        """
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __dir__(self):
        """
        Enhance attribute autocompletion by including dictionary keys.
        """
        # Combine default attributes with current dictionary keys
        return super().__dir__() + list(self.keys())

