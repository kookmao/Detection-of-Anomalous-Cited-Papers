import abc
from typing import Optional, Any, Dict

class method(abc.ABC):
    """Abstract base class for methods"""
    def __init__(self, mName: Optional[str] = None, mDescription: Optional[str] = None):
        self.method_name: Optional[str] = mName
        self.method_description: Optional[str] = mDescription
        self.data: Optional[Dict] = None
        # Time tracking
        self.method_start_time: Optional[float] = None
        self.method_stop_time: Optional[float] = None
        self.method_running_time: Optional[float] = None
        self.method_training_time: Optional[float] = None
        self.method_testing_time: Optional[float] = None

    @abc.abstractmethod
    def run(self, trainData: Optional[Any] = None, 
           trainLabel: Optional[Any] = None, 
           testData: Optional[Any] = None) -> Any:
        pass