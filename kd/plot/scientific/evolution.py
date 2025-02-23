from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip, ImageSequenceClip
import imageio.v2 as imageio
from ..core.base import BasePlot
import shutil

class EvolutionVisBase(BasePlot):
    """Base class for evolution visualization"""
    
    def __init__(self, figsize=(10, 8), dpi=100):
        super().__init__(figsize=figsize)
        self.dpi = dpi  # Store dpi for later use
        
        # Create figure
        self.fig = plt.figure(figsize=self.figsize)
        
        # Create subplots
        self.gs = plt.GridSpec(2, 2, figure=self.fig)
        self.axes = {
            'population': self.fig.add_subplot(self.gs[0, :]),
            'best': self.fig.add_subplot(self.gs[1, 0]),
            'stats': self.fig.add_subplot(self.gs[1, 1])
        }
        
    def clear(self):
        """Clear all axes"""
        for ax in self.axes.values():
            ax.clear()
            
    def save(self):
        """Override save method to use stored dpi value"""
        if self.save_path and self.fig:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # Save figure with stored dpi
            self.fig.savefig(self.save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to: {self.save_path}")
        
    def plot(self, data):
        """Plot single generation state.
        
        This is the implementation of the abstract method from BasePlot.
        For evolution visualization, we use more specific methods in subclasses.
        
        Args:
            data: Generation data
        """
        self._plot_generation(data)
        
    def _plot_generation(self, data):
        """Plot single generation state
        
        Args:
            data: Generation data containing:
                - population: List of individuals
                - best_individual: Best individual data
                - stats: Dictionary of statistics
                - generation: Generation number
        """
        # Clear all axes
        self.clear()
            
        # Plot each component
        self._plot_population(data.population, data.generation)
        self._plot_best_individual(data.best_individual)
        self._plot_stats(data.stats)
        
        # Adjust layout
        self.fig.tight_layout()
        
    def _plot_population(self, population, generation):
        """Plot population distribution
        
        Args:
            population: List of individuals
            generation: Current generation number
        """
        ax = self.axes['population']
        
        # Extract fitness values
        fitness_values = [ind.fitness for ind in population]
        
        # Create violin plot
        ax.violinplot(fitness_values, positions=[generation])
        
        # Add scatter points for individual fitness
        ax.scatter([generation] * len(fitness_values), 
                  fitness_values,
                  alpha=0.4, 
                  color='blue')
        
        ax.set_title(f'Population Distribution (Gen {generation})')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid(True, linestyle=':')
        
    def _plot_best_individual(self, individual):
        """Plot best individual
        
        Args:
            individual: Best individual data containing:
                - equation: Equation string
                - complexity: Complexity measure
                - fitness: Fitness value
        """
        ax = self.axes['best']
        
        # Clear previous text
        ax.clear()
        ax.axis('off')
        
        # Create info text
        info = [
            f"Best Individual:",
            f"Equation: {individual.equation}",
            f"Fitness: {individual.fitness:.6f}",
            f"Complexity: {individual.complexity}"
        ]
        
        # Display text
        ax.text(0.05, 0.95, '\n'.join(info),
                transform=ax.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         alpha=0.8))
        
    def _plot_stats(self, stats):
        """Plot statistics
        
        Args:
            stats: Dictionary containing:
                - mean_fitness: Mean fitness
                - std_fitness: Fitness standard deviation
                - diversity: Population diversity
                - elapsed_time: Time elapsed
        """
        ax = self.axes['stats']
        
        # Clear previous text
        ax.clear()
        ax.axis('off')
        
        # Create stats text
        stats_info = [
            f"Statistics:",
            f"Mean Fitness: {stats['mean_fitness']:.6f}",
            f"Std Fitness: {stats['std_fitness']:.6f}",
            f"Diversity: {stats['diversity']:.6f}",
            f"Time: {stats['elapsed_time']:.2f}s"
        ]
        
        # Display text
        ax.text(0.05, 0.95, '\n'.join(stats_info),
                transform=ax.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         alpha=0.8))

class EvolutionAnimation(EvolutionVisBase):
    """Real-time animation for evolution visualization"""
    
    def __init__(self, figsize=(10, 8), dpi=100, interval=100):
        super().__init__(figsize=figsize, dpi=dpi)
        self.interval = interval
        self.anim = None
        
    def animate(self, evolution_data):
        """Create and display animation
        
        Args:
            evolution_data: List/Iterator of generation data
        """
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=evolution_data,
            interval=self.interval,
            blit=True
        )
        
    def _update_frame(self, frame_data):
        """Update single animation frame"""
        self.clear()
        self._plot_generation(frame_data)
        return self.fig.get_axes()
        
    def add_controls(self):
        """Add animation control widgets"""
        # TODO: Implement play/pause, speed controls
        pass

class EvolutionSnapshot(EvolutionVisBase):
    """Frame sequence generator for evolution visualization"""
    
    def __init__(self, figsize=(10, 8), dpi=300,
                 temp_dir=".evolution_temp",
                 output_format="mp4",
                 fps=30,
                 desired_duration=15,  # 新增: 期望的视频时长(秒)
                 cleanup_temp=True):
        """
        Args:
            figsize: Figure size
            dpi: DPI for saved images
            temp_dir: Directory for temporary frame files
            output_format: Output format, one of ['mp4', 'gif', 'frames']
            fps: Default frames per second for video
            desired_duration: Desired duration in seconds for video/gif
            cleanup_temp: Whether to cleanup temp files after processing
        """
        super().__init__(figsize=figsize, dpi=dpi)
        self.temp_dir = Path(temp_dir)
        self.output_format = output_format
        self.fps = fps
        self.desired_duration = desired_duration
        self.cleanup_temp = cleanup_temp
        
    def save_evolution(self, evolution_data, output_path=None):
        """Save evolution visualization
        
        Args:
            evolution_data: List/Iterator of generation data
            output_path: Output file/directory path. If None, use default name.
        """
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Save frames
        frame_files = self._save_frames(evolution_data)
        
        # Check if we have any frames
        if not frame_files:
            raise ValueError("No frames were generated. Check if evolution_data is empty.")
            
        # Process based on format
        if output_path is None:
            output_path = f"evolution.{self.output_format}"
            
        if self.output_format == "mp4":
            self._make_video(frame_files, output_path)
        elif self.output_format == "gif":
            self._make_gif(frame_files, output_path)
        elif self.output_format == "frames":
            if not output_path.endswith("/"):
                output_path += "/"
            output_path = self._save_frames_to_dir(frame_files, output_path)
            
        # Cleanup if needed
        if self.cleanup_temp:
            self._cleanup(frame_files)
            
        return output_path
        
    def _save_frame(self, frame_data, frame_num):
        """Save single frame
        
        Args:
            frame_data: Generation data for this frame
            frame_num: Frame number
            
        Returns:
            Path: Path to saved frame file
        """
        # Clear and plot
        self.clear()
        self._plot_generation(frame_data)
        
        # Save frame with lower DPI for better performance
        frame_path = self.temp_dir / f"frame_{frame_num:05d}.png"
        self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')  # Lower DPI for temp files
        return frame_path
        
    def _save_frames(self, evolution_data):
        """Save all frames using parallel processing
        
        Args:
            evolution_data: List/Iterator of generation data
            
        Returns:
            list: List of paths to saved frame files
        """
        frame_files = []
        
        # Convert to list to avoid iterator issues
        evolution_data = list(evolution_data)
        
        # Calculate optimal chunk size based on CPU count and data size
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        total_frames = len(evolution_data)
        
        # For very large datasets, use more workers
        if total_frames > 1000:
            max_workers = cpu_count * 2
        else:
            max_workers = cpu_count
            
        # Adjust chunk size based on total frames
        chunk_size = max(1, min(
            total_frames // (max_workers * 2),  # 2 chunks per worker
            20  # Cap chunk size for memory management
        ))
        
        # Split data into chunks
        chunks = [evolution_data[i:i + chunk_size] 
                 for i in range(0, total_frames, chunk_size)]
        
        # Create a pool of workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks at once
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                for i, data in enumerate(chunk):
                    frame_num = chunk_idx * chunk_size + i
                    future = executor.submit(self._save_frame, data, frame_num)
                    futures.append(future)
                    
            # Collect results as they complete
            import concurrent.futures
            for future in concurrent.futures.as_completed(futures):
                frame_files.append(future.result())
                
        return sorted(frame_files)
                
    def _make_video(self, frame_files, output_path):
        """Create video from frames
        
        Args:
            frame_files: List of paths to frame files
            output_path: Output video path
            
        Note:
            Total duration = len(frame_files) / fps
            例如: 100帧, fps=10 -> 10秒视频
        """
        # 根据期望时长计算 fps
        n_frames = len(frame_files)
        fps = int(n_frames / self.desired_duration)  # 确保 fps 至少为 1
        fps = max(1, min(fps, 30))  # 限制 fps 在 1-30 之间
        
        clip = ImageSequenceClip([str(f) for f in frame_files],
                               fps=fps)
        clip.write_videofile(output_path, codec='libx264')
        
    def _make_gif(self, frame_files, output_path):
        """Create GIF from frames
        
        Args:
            frame_files: List of paths to frame files
            output_path: Output GIF path
        """
        # 读取所有帧
        frames = []
        for frame_file in frame_files:
            frames.append(imageio.imread(frame_file))
        
        # 计算合适的帧率
        n_frames = len(frames)
        fps = int(n_frames / self.desired_duration)  # 确保 fps 至少为 1
        fps = max(1, min(fps, 30))  # 限制 fps 在 1-30 之间
        
        # 保存GIF，duration单位是毫秒
        duration = 1000 / fps  # 转换fps为毫秒间隔
        imageio.mimsave(output_path, frames, duration=duration)
        
    def _save_frames_to_dir(self, frame_files, output_dir):
        """Save frames to directory
        
        Args:
            frame_files: List of paths to frame files
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 使用格式化的文件名，确保正确的顺序
        for i, frame_file in enumerate(frame_files):
            new_path = output_path / f"evolution_frame_{i:05d}.png"
            shutil.copy2(frame_file, new_path)  # 使用copy2保留元数据
            
        return output_path
            
    def _cleanup(self, frame_files):
        """Clean up temporary files
        
        Args:
            frame_files: List of paths to frame files
        """
        # Remove frame files
        for frame_file in frame_files:
            try:
                if frame_file.exists():
                    frame_file.unlink()
            except OSError:
                pass  # Ignore errors
                
        # Try to remove temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass  # Ignore errors 