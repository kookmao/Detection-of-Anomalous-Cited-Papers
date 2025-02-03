from codes.base_class.setting import setting
from typing import Optional, Any

class Settings(setting):
    """Settings class for managing dataset and method interactions"""
    
    def __init__(self, sName: Optional[str] = None, sDescription: Optional[str] = None):
        super().__init__(sName, sDescription)
        self._initialized = False
        self._validate_state()

    def _validate_state(self) -> None:
        """Validate internal state"""
        if self.dataset is not None and not hasattr(self.dataset, 'load'):
            raise AttributeError("Dataset must implement load() method")
        if self.method is not None and not hasattr(self.method, 'run'):
            raise AttributeError("Method must implement run() method")

    def prepare(self, sDataset: Any, sMethod: Any) -> None:
        """Prepare settings with dataset and method validation"""
        if sDataset is None:
            raise ValueError("Dataset cannot be None")
        if sMethod is None:
            raise ValueError("Method cannot be None")
            
        # Verify dataset interface
        if not hasattr(sDataset, 'load'):
            raise AttributeError("Dataset must implement load() method")
        
        # Verify method interface
        if not hasattr(sMethod, 'run'):
            raise AttributeError("Method must implement run() method")
        
        self.dataset = sDataset
        self.method = sMethod
        self._initialized = True
        self._validate_state()

    def run(self) -> Optional[Any]:
        """Execute the prepared settings with proper error handling"""
        if not self._initialized:
            raise RuntimeError("Settings not initialized. Call prepare() first")
            
        try:
            # Load dataset
            loaded_data = self.dataset.load()
            if loaded_data is None:
                raise ValueError("Dataset load() returned None")
                
            # Set method data
            self.method.data = loaded_data
            
            # Run method
            result = self.method.run()
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error during execution: {str(e)}") from e

    def __str__(self) -> str:
        """String representation of settings state"""
        status = "initialized" if self._initialized else "not initialized"
        dataset_name = getattr(self.dataset, 'dataset_name', 'None')
        method_name = getattr(self.method, 'method_name', 'None')
        return f"Settings({status}, dataset={dataset_name}, method={method_name})"

    def reset(self) -> None:
        """Reset settings to initial state"""
        self.dataset = None
        self.method = None
        self._initialized = False