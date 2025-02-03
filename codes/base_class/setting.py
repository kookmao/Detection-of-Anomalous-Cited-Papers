import abc
from typing import Optional, Any
from codes.base_class.dataset import dataset
from codes.base_class.method import method
from codes.base_class.result import result
from codes.base_class.evaluate import evaluate

class setting(abc.ABC):
    """Abstract base class for settings"""
    def __init__(self, sName: Optional[str] = None, sDescription: Optional[str] = None):
        self.setting_name: Optional[str] = sName
        self.setting_description: Optional[str] = sDescription
        self.dataset: Optional[dataset] = None
        self.method: Optional[method] = None
        self.result: Optional[result] = None
        self.evaluate: Optional[evaluate] = None

    def prepare(self, sDataset: dataset, sMethod: method) -> None:
        """Prepare settings with type validation"""
        if not isinstance(sDataset, dataset):
            raise TypeError("sDataset must be an instance of dataset")
        if not isinstance(sMethod, method):
            raise TypeError("sMethod must be an instance of method")
        self.dataset = sDataset
        self.method = sMethod

    @abc.abstractmethod
    def run(self) -> Any:
        pass