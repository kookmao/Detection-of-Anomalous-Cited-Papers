import abc
from typing import Optional, Any, Dict

class evaluate(abc.ABC):
    """Abstract base class for evaluation"""
    def __init__(self, eName: Optional[str] = None, eDescription: Optional[str] = None):
        self.evaluate_name: Optional[str] = eName
        self.evaluate_description: Optional[str] = eDescription
        self.data: Optional[Dict] = None

    @abc.abstractmethod
    def evaluate(self) -> Any:
        pass