import dataclasses
import copy

@dataclasses.dataclass
class DatasetInfo:
    """
    Class representing the metadata for a dataset.

    This class holds information about the dataset such as its description, citation, homepage, and license.

    Attributes:
        description (str): A short description of the dataset.
        citation (str): Citation information for the dataset (e.g., paper citation).
        homepage (str): The URL to the dataset's homepage.
        license (str): The license information of the dataset (e.g., MIT, Apache 2.0).
    """
    
    # Attribute definitions with default empty string values
    description: str = dataclasses.field(default_factory=str)  # Description of the dataset
    citation: str = dataclasses.field(default_factory=str)      # Citation information for the dataset
    homepage: str = dataclasses.field(default_factory=str)      # URL to the dataset's homepage
    license: str = dataclasses.field(default_factory=str)       # License under which the dataset is distributed

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        """
        Create a `DatasetInfo` instance from a dictionary.

        This method filters the dictionary to only include keys that match the attribute names
        of the `DatasetInfo` class and uses them to instantiate the class.

        Args:
            dataset_info_dict (dict): A dictionary containing dataset metadata, where the keys correspond
                                      to the attribute names of the class.

        Returns:
            DatasetInfo: A `DatasetInfo` instance populated with data from the dictionary.
        """
        # Get the set of field names defined in the class
        field_names = {f.name for f in dataclasses.fields(cls)}
        
        # Filter the dictionary and create a new instance by unpacking the matching key-value pairs
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def copy(self) -> "DatasetInfo":
        """
        Create a deep copy of the current `DatasetInfo` instance.

        This method ensures that all fields in the instance, including any mutable objects, are deeply copied.
        
        Returns:
            DatasetInfo: A new `DatasetInfo` instance with the same data as the current instance,
                         but independent of it.
        """
        # Create a deep copy of the instance's dictionary and instantiate a new object with the copied data
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

