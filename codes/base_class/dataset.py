import abc
from typing import Dict, Optional


class dataset(abc.ABC):
    """Abstract base class for datasets"""
    def __init__(self, dName: Optional[str] = None, dDescription: Optional[str] = None):
        self.dataset_name: Optional[str] = dName
        self.dataset_descrition: Optional[str] = dDescription  # Keep typo for compatibility
        self.dataset_source_folder_path: Optional[str] = None
        self.dataset_source_file_name: Optional[str] = None
        self.data: Optional[Dict] = None

    @abc.abstractmethod
    def load(self) -> Dict:
        """Load dataset"""
        pass