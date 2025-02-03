from typing import Optional, Any, Dict
from codes.base_class.setting import setting

class Settings(setting):
    """
    Settings class for experiment configuration and execution
    """
    fold: Optional[int] = None
    
    def prepare(self, sDataset: Any, sMethod: Any) -> None:
        """
        Override prepare to add validation
        """
        if sDataset is None or sMethod is None:
            raise ValueError("Dataset and method must be provided")
            
        super().prepare(sDataset, sMethod)
    
    def run(self) -> Optional[Dict]:
        """
        Execute the experiment pipeline
        """
        if self.dataset is None:
            raise ValueError("Dataset not initialized. Call prepare() first")
        if self.method is None:
            raise ValueError("Method not initialized. Call prepare() first")
            
        try:
            # Load dataset
            loaded_data = self.dataset.load()
            if loaded_data is None:
                raise ValueError("Dataset loading failed")
                
            # Initialize method with data
            self.method.data = loaded_data
            
            # Execute method
            learned_result = self.method.run()
            
            return learned_result
            
        except Exception as e:
            print(f"Error in Settings.run(): {str(e)}")
            raise