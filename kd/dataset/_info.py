import dataclasses
import copy
from typing import Optional
from importlib import resources
from pathlib import Path

@dataclasses.dataclass
class DatasetInfo:
    """
    Class representing the metadata for a dataset.

    This class holds information about the dataset such as its description, citation, homepage, license,
    author information, and keywords.

    Attributes:
        description (str): A short description of the dataset.
        citation (str): Citation information for the dataset (e.g., paper citation).
        homepage (str): The URL to the dataset's homepage.
        license (str): The license information of the dataset (e.g., MIT, Apache 2.0).
        author (Optional[str]): The author(s) of the dataset (optional).
        keywords (Optional[list]): List of keywords relevant to the dataset (optional).
        version (Optional[str]): Version of the dataset (optional).
    """
    
    # Attribute definitions with default empty string values
    description: str = dataclasses.field(default_factory=str)  # Description of the dataset
    citation: str = dataclasses.field(default_factory=str)      # Citation information for the dataset
    homepage: str = dataclasses.field(default_factory=str)      # URL to the dataset's homepage
    license: str = dataclasses.field(default_factory=str)       # License under which the dataset is distributed
    author: Optional[str] = dataclasses.field(default_factory=str)  # Author(s) of the dataset (optional)
    keywords: Optional[list] = dataclasses.field(default_factory=list)  # Keywords related to the dataset
    version: Optional[str] = dataclasses.field(default_factory=str)  # Dataset version (optional)

    def __post_init__(self):
        """Ensure the keywords are always a list if it's not None."""
        if isinstance(self.keywords, str):  # If the keywords are provided as a comma-separated string, split them
            self.keywords = [kw.strip() for kw in self.keywords.split(',')]
        elif self.keywords is None:
            self.keywords = []

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

    def load_descr(descr_file_name, *, encoding="utf-8"):
        """
        Load and read the content of a specified description file, supporting both absolute and relative file paths.
        
        Parameters:
            descr_file_name (str): The path of the description file to load. Can be either an absolute or relative path.
            encoding (str, default="utf-8"): The encoding used to read the file.
        
        Returns:
            str: The content of the description file.
        
        This function reads the content of a description file based on the provided path. If a relative path is given,
        it is converted into an absolute path based on the current working directory.
        """
        # Convert the description file path to a Path object
        file_path = Path(descr_file_name)

        # If it's a relative path, convert it to an absolute path
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path  # Combine current working directory with relative path
            
        # path = resources.files(descr_module) / descr_file_name

        # Ensure the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Description file '{descr_file_name}' not found at the specified path.")

        # Read the content of the file and return it
        return file_path.read_text(encoding=encoding)


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

    def update(self, **kwargs) -> None:
        """
        Update the attributes of the current `DatasetInfo` instance with new values.

        This method allows for dynamic updates to the dataset's metadata by providing keyword arguments.

        Args:
            **kwargs: Key-value pairs representing attributes to update in the instance.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def to_dict(self) -> dict:
        """
        Convert the `DatasetInfo` instance to a dictionary.

        Returns:
            dict: A dictionary representation of the `DatasetInfo` instance.
        """
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the `DatasetInfo` instance.
        
        This provides a human-readable output for the object, listing its attributes and values.
        """
        return f"DatasetInfo(description={self.description}, citation={self.citation}, homepage={self.homepage}, " \
               f"license={self.license}, author={self.author}, keywords={self.keywords}, version={self.version})"
    
    def __eq__(self, other) -> bool:
        """
        Check if two `DatasetInfo` instances are equal.
        
        This compares the values of all attributes to determine equality.

        Args:
            other (DatasetInfo): The other `DatasetInfo` instance to compare with.

        Returns:
            bool: True if the two instances have the same attribute values, otherwise False.
        """
        if not isinstance(other, DatasetInfo):
            return False
        return self.to_dict() == other.to_dict()
