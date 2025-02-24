"""Plot system configuration.

统一管理 plot 系统的配置，包括输出路径等
"""

from pathlib import Path
import os

class PlotConfig:
    """Plot system configuration.
    
    统一管理 plot 系统的配置，包括输出路径等
    """
    
    # Base directories
    BASE_OUTPUT_DIR = "plot_output"
    TEMP_DIR = "temp"
    
    # Sub directories
    EVOLUTION_DIR = "evolution"
    RESIDUAL_DIR = "residual"
    TRAINING_DIR = "training"
    EQUATION_DIR = "equation"
    
    @classmethod
    def get_output_dir(cls, category: str = None) -> Path:
        """Get output directory path.
        
        Args:
            category: Directory category (evolution/residual/training/equation)
            
        Returns:
            Path object for the requested directory
        """
        base_dir = Path(cls.BASE_OUTPUT_DIR)
        if category:
            base_dir = base_dir / category
            
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
        
    @classmethod
    def get_temp_dir(cls) -> Path:
        """Get temporary directory path.
        
        Returns:
            Path object for temporary directory
        """
        temp_dir = Path(cls.BASE_OUTPUT_DIR) / cls.TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir 