import abc
from typing import Optional, Any, Dict

class result(abc.ABC):
    """Abstract base class for results"""
    def __init__(self, rName: Optional[str] = None, rType: Optional[str] = None):
        self.result_name: Optional[str] = rName
        self.result_description: Optional[str] = rType
        self.data: Optional[Dict] = None
        self.result_destination_folder_path: Optional[str] = None
        self.result_destination_file_name: Optional[str] = None

    @abc.abstractmethod
    def save(self) -> None:
        pass

    @abc.abstractmethod
    def load(self) -> Dict:
        pass