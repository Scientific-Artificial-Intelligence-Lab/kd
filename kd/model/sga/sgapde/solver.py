"""
Main solver class for the SGA-PDE system.
"""

import numpy as np
import copy
import random
import warnings

warnings.filterwarnings('ignore')

class SGAPDE_Solver:
    """Main solver class for SGA-PDE discovery."""
    
    def __init__(self, config):
        """
        Initialize the solver with configuration.
        
        Args:
            config: SolverConfig object containing all parameters
        """
        self.config = config
        
        # Set random seeds
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Initialize PDE and error libraries
        self.pde_lib = []
        self.err_lib = []
        
    def run(self, context):
        """
        Run the SGA algorithm to discover PDEs.
        
        Args:
            context: ProblemContext object containing problem data
            
        Returns:
            tuple: (best_pde_string, best_score)
        """
        # Import necessary modules with context
        from sgapde.pde import PDE, evaluate_mse
        
        # Make context available globally for the old code
        # This is a temporary measure during refactoring
        # self._inject_context_to_modules(context)
        
        # Initialize the SGA algorithm
        sga = SGA(
            context=context,
            num=self.config.num,
            depth=self.config.depth,
            width=self.config.width,
            p_var=self.config.p_var,
            p_mute=self.config.p_mute,
            p_rep=self.config.p_rep,
            p_cro=self.config.p_cro,
            pde_lib=self.pde_lib,
            err_lib=self.err_lib
        )
        
        # Run the genetic algorithm
        best_eq, best_mse = sga.run(self.config.sga_run)
        
        # Return the best PDE and its score
        return best_eq.visualize(), best_mse

class SGA:
    """Genetic Algorithm for PDE discovery (refactored from original sga.py)."""
    
    def __init__(self, context, num, depth, width, p_var, p_mute, p_rep, p_cro, pde_lib, err_lib):
        """
        Initialize the SGA algorithm.
        
        Args:
            context: ProblemContext object
            num: Number of PDEs in the pool
            depth: Maximum depth of each PDE term
            width: Maximum number of terms in each PDE
            p_var: Probability of node being variable instead of operator
            p_rep: Probability of replacing a term
            p_mute: Mutation probability for each node
            p_cro: Crossover probability between PDEs
            pde_lib: PDE library list
            err_lib: Error library list
        """
        from sgapde.pde import PDE, evaluate_mse
        
        self.context = context
        self.num = num
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.p_rep = p_rep
        self.eqs = []
        self.mses = []
        self.ratio = 1
        self.repeat_cross = 0
        self.repeat_change = 0
        self.pde_lib = pde_lib
        self.err_lib = err_lib
        
        print('Creating the original pdes in the pool ...')
        for i in range(num * self.ratio):
            a_pde = PDE(self.context, depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde, self.context)
            self.pde_lib.append(a_pde)
            self.err_lib.append((a_err, a_w))
            
            while a_err < -100 or a_err == np.inf:
                print(a_err)
                a_pde = PDE(self.context, depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde, self.context)
                self.pde_lib.append(a_pde)
                self.err_lib.append((a_err, a_w))
                
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():', a_pde.visualize())
            print('evaluate_aic:', a_err)
            self.eqs.append(a_pde)
            self.mses.append(a_err)
        
        # Sort by MSE
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        self.mses, self.eqs = self.mses[0:num], self.eqs[0:num]
    
    def run(self, gen=100):
        """
        Run the genetic algorithm for specified generations.
        
        Args:
            gen: Number of generations to run
            
        Returns:
            tuple: (best_equation, best_mse)
        """
        for i in range(1, gen + 1):
            self.cross_over(self.p_cro)
            self.change(self.p_mute, self.p_rep)
            best_eq, best_mse = self.the_best()
            print('{} generation best_aic & best Eq: {}, {}'.format(i, best_mse, best_eq.visualize()))
            print('best concise Eq: {}'.format(best_eq.concise_visualize(self.context)))
            if best_mse < 0:
                print('We are close to the answer, pay attention')
            print('{} generation repeat cross over {} times and mutation {} times'.format(
                i, self.repeat_cross, self.repeat_change))
            self.repeat_cross, self.repeat_change = 0, 0
        
        return self.the_best()
    
    def the_best(self):
        """Get the best equation and its MSE."""
        argmin = np.argmin(self.mses)
        return self.eqs[argmin], self.mses[argmin]
    
    def cross_over(self, percentage=0.5):
        """Perform crossover operation on the population."""
        from sgapde.pde import evaluate_mse
        
        def cross_individual(pde1, pde2):
            new_pde1, new_pde2 = copy.deepcopy(pde1), copy.deepcopy(pde2)
            w1, w2 = len(pde1.elements), len(pde2.elements)
            ix1, ix2 = np.random.randint(w1), np.random.randint(w2)
            new_pde1.elements[ix1] = pde2.elements[ix2]
            new_pde2.elements[ix2] = pde1.elements[ix1]
            return new_pde1, new_pde2
        
        num_ix = int(self.num * percentage)
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        copy_mses, copy_eqs = self.mses[0:num_ix], self.eqs[0:num_ix]
        
        new_eqs, new_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        reo_eqs, reo_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        random.shuffle(reo_mse)
        random.shuffle(reo_eqs)
        
        for a, b in zip(new_eqs, reo_eqs):
            new_a, new_b = cross_individual(a, b)
            if new_a.visualize() in self.pde_lib:
                self.repeat_cross += 1
            else:
                a_err, a_w = evaluate_mse(new_a, self.context)
                self.pde_lib.append(new_a.visualize())
                self.err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_a)
            
            if new_b.visualize() in self.pde_lib:
                self.repeat_cross += 1
            else:
                b_err, b_w = evaluate_mse(new_b, self.context)
                self.pde_lib.append(new_b.visualize())
                self.err_lib.append((b_err, b_w))
                self.mses.append(b_err)
                self.eqs.append(new_b)
        
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num]
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
    
    def change(self, p_mute=0.05, p_rep=0.3):
        """Perform mutation and replacement operations."""
        from sgapde.pde import evaluate_mse
        
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        
        for i in range(self.num):
            if i < 1:  # Keep the best sample unchanged
                continue
            
            new_eqs[i].mutate(p_mute)
            replace_or_not = np.random.choice([False, True], p=([1 - p_rep, p_rep]))
            if replace_or_not:
                new_eqs[i].replace()
            
            if new_eqs[i].visualize() in self.pde_lib:
                self.repeat_change += 1
            else:
                a_err, a_w = evaluate_mse(new_eqs[i], self.context)
                self.pde_lib.append(new_eqs[i].visualize())
                self.err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_eqs[i])
        
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num]
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
