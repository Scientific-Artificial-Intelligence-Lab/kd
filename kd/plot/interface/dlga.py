from ..scientific.evolution import EvolutionAnimation, EvolutionSnapshot

class DLGAPlotter:
    """Interface for DLGA plotting functionalities"""
    
    def plot_evolution(self, evolution_data, mode="animation", **kwargs):
        """Plot evolution process
        
        Args:
            evolution_data: List/Iterator of generation data
            mode: Visualization mode, one of ['animation', 'snapshot']
            **kwargs: Additional arguments passed to specific plotter
                For animation mode:
                    - interval: Animation interval in ms (default: 100)
                    - figsize: Figure size (default: (10,8))
                    - dpi: DPI for display (default: 100)
                
                For snapshot mode:
                    - output_format: Output format ['mp4','gif','frames'] (default: 'mp4')
                    - output_path: Output file/directory path (default: None)
                    - temp_dir: Directory for temporary files (default: '.evolution_temp')
                    - fps: Frames per second for video (default: 30)
                    - cleanup_temp: Whether to cleanup temp files (default: True)
                    - figsize: Figure size (default: (10,8))
                    - dpi: DPI for saved images (default: 300)
                    - desired_duration: Desired duration in seconds (default: 15)
        
        Returns:
            Animation object if mode is 'animation'
            Output path if mode is 'snapshot'
        """
        if mode == "animation":
            plotter = EvolutionAnimation(
                figsize=kwargs.get("figsize", (10,8)),
                dpi=kwargs.get("dpi", 100),
                interval=kwargs.get("interval", 100)
            )
            plotter.animate(evolution_data)
            return plotter.anim
            
        elif mode == "snapshot":
            plotter = EvolutionSnapshot(
                figsize=kwargs.get("figsize", (10,8)),
                dpi=kwargs.get("dpi", 300),
                temp_dir=kwargs.get("temp_dir", ".evolution_temp"),
                output_format=kwargs.get("output_format", "mp4"),
                fps=kwargs.get("fps", 30),
                cleanup_temp=kwargs.get("cleanup_temp", True),
                desired_duration=kwargs.get("desired_duration", 15)
            )
            output_path = kwargs.get("output_path", None)
            plotter.save_evolution(evolution_data, output_path)
            return output_path
            
        else:
            raise ValueError(f"Unknown mode: {mode}") 