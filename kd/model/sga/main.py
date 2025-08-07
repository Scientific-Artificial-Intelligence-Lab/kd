"""
Main entry point for the SGA-PDE solver.
This demonstrates how to use the refactored library.
"""

import sys
import warnings
from sgapde.config import SolverConfig
from sgapde.context import ProblemContext
from sgapde.solver import SGAPDE_Solver
import sgapde.visualizer as visualizer

warnings.filterwarnings('ignore')


class Logger:
    """Logger to write output to both console and file."""
    
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == "__main__":
    # Set up logging
    sys.stdout = Logger('notes.log', sys.stdout)
    sys.stderr = Logger('notes.log', sys.stderr)
    
    print("SGA-PDE: Symbolic Genetic Algorithm for PDE Discovery")
    
    # 1. Set up configuration
    print("\n1. Setting up configuration...")
    config = SolverConfig(
        problem_name='chafee-infante',  # Problem to solve
        num=20,  # Number of PDEs in the pool
        depth=4,  # Maximum depth of each PDE term
        width=5,  # Maximum number of terms in each PDE
        p_var=0.5,  # Probability of node being variable
        p_mute=0.3,  # Mutation probability
        p_cro=0.5,  # Crossover probability
        sga_run=3,  # Number of generations
        seed=0,  # Random seed for reproducibility
        simple_mode=True,  # Use simple mode
        use_autograd=False  # Use autograd for derivatives
    )
    
    print(f"\tProblem: {config.problem_name}")
    print(f"\tPool size: {config.num}")
    print(f"\tMax depth: {config.depth}")
    print(f"\tMax width: {config.width}")
    print(f"\tGenerations: {config.sga_run}")

    # 2. Prepare data and context
    print("\n2. Preparing problem context...")
    context = ProblemContext(config)
    print(f"\tData shape: {context.u.shape}")
    print(f"\tTime points: {context.t.shape[1]}")
    print(f"\tSpace points: {context.x.shape[0]}")

    # 3. Visualize data
    if config.use_metadata and config.show_metadata_diagnostics:
        visualizer.plot_metadata_diagnostics(context)

    visualizer.plot_figures(context, config)

    # 4. Create and run solver
    print("\n3. Creating solver and running genetic algorithm...")
    print("=" * 80)
    solver = SGAPDE_Solver(config)
    best_pde, best_score = solver.run(context)
    
    # 5. Print results
    print("=" * 80)
    print("\nDiscovery finished!")
    print(f"Best PDE found: {best_pde}")
    print(f"AIC Score: {best_score}")

