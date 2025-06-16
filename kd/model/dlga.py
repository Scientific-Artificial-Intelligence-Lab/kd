"""Deep Learning Genetic Algorithm (DLGA) for PDE discovery.

This module implements a hybrid approach combining neural networks and genetic algorithms
to discover governing equations of PDEs. It uses neural networks to fit the data and
genetic algorithms to search for the symbolic form of equations.
"""

import os
import random
from abc import ABCMeta, abstractmethod
import torch.utils.data
from ..base import BaseEstimator
from ..utils.utils_GA import *
import numpy as np
import heapq
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from ..viz import equation_renderer

class BaseGa(BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for genetic algorithm based models."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit model to data."""

    def predict(self, X):
        """Make predictions."""
        pass


class DLGA(BaseGa):
    """Deep Learning Genetic Algorithm for PDE discovery.
    
    This class combines neural networks with genetic algorithms to discover PDEs.
    The neural network fits the data while the genetic algorithm searches for
    the symbolic form of the governing equations.
    
    Attributes:
        max_length (int): Maximum length of gene modules.
        partial_prob (float): Probability for partial gene generation.
        genes_prob (float): Probability for gene generation.
        mutate_rate (float): Mutation rate for genetic algorithm.
        delete_rate (float): Rate for deleting gene modules.
        add_rate (float): Rate for adding gene modules.
        pop_size (int): Population size for genetic algorithm.
        n_generations (int): Number of generations to evolve.
        train_ratio (float): Ratio of training data.
        valid_ratio (float): Ratio of validation data.
        device (torch.device): Device to use for computation.
        Net (NN): Neural network model.
        epi (float): Penalty coefficient for equation length.
        fitness_history (list): History of best fitness values.
        train_loss_history (list): History of training loss values.
        val_loss_history (list): History of validation loss values.
        metadata (dict): Metadata for equation terms.
        evolution_history (list): History of evolution data.
        complexity_history (list): History of complexity data.
    """
    
    _parameter: dict = {}

    def __init__(self, epi, input_dim, max_iter=50000):
        """Initialize DLGA model.
        
        Args:
            epi (float): Penalty coefficient for equation length.
            input_dim (int): Input dimension for neural network.
        """
        super().__init__()
        self.max_length = 5
        self.partial_prob = 0.6
        self.genes_prob = 0.6
        self.mutate_rate = 0.4
        self.delete_rate = 0.5
        self.add_rate = 0.4
        self.pop_size = 400
        self.n_generations = 100
        self.train_ratio = 0.8
        self.valid_ratio = 0.2
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.Net = NN(
            Num_Hidden_Layers=5,
            Neurons_Per_Layer=50,
            Input_Dim=input_dim,
            Output_Dim=1,
            Data_Type=torch.float32,
            Device=self.device,
            Activation_Function="Sin",
            Batch_Norm=False,
        )

        self.max_iter = max_iter
        self.eq_latex = ""
        self.epi = epi
        self.fitness_history = []  # Track best fitness per generation
        self.train_loss_history = []  # 训练损失历史
        self.val_loss_history = []    # 验证损失历史
        self.metadata = {}  # 新增元数据存储
        self.evolution_history = []  # 存储完整的进化历史数据
        self.complexity_history = []  # 存储复杂度历史

    def train_NN(self, X, y):
        """Train neural network on data.
        
        Args:
            X: Input features.
            y: Target values.
            
        Returns:
            tuple: (trained network, best epoch)
        """
        # Shuffle data
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)

        # Split data
        X_train = X[0 : int(X.shape[0] * self.train_ratio)]
        y_train = y[0 : int(X.shape[0] * self.train_ratio)]
        X_valid = X[
            int(X.shape[0] * self.train_ratio) : int(X.shape[0] * self.train_ratio)
            + int(X.shape[0] * self.valid_ratio)
        ]
        y_valid = y[
            int(X.shape[0] * self.train_ratio) : int(X.shape[0] * self.train_ratio)
            + int(X.shape[0] * self.valid_ratio)
        ]

        # Convert to tensors
        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        X_valid = torch.from_numpy(X_valid.astype(np.float32)).to(self.device)

        # Setup optimizer
        NN_optimizer = torch.optim.Adam([{"params": self.Net.parameters()}])
        MSELoss = torch.nn.MSELoss()
        validate_error = []

        # Create model directory
        try:
            os.makedirs("model_save/")
        except OSError:
            pass

        print("===============train Net=================")

        # Training loop
        for iter in range(self.max_iter):
            NN_optimizer.zero_grad()
            prediction = self.Net(X_train)
            prediction_validate = self.Net(X_valid)
            loss = MSELoss(prediction, y_train.view(-1, 1))
            loss_validate = np.mean(
                (prediction_validate.detach().cpu().numpy() - y_valid) ** 2
            )
            loss.backward()
            NN_optimizer.step()

            if (iter + 1) % 500 == 0:
                validate_error.append(loss_validate)
                torch.save(self.Net.state_dict(), f"model_save/Net_{iter + 1}.pkl")
                print(
                    "iter_num: %d      loss: %.8f    loss_validate: %.8f"
                    % (iter + 1, loss, loss_validate)
                )

            self.train_loss_history.append(float(loss))
            self.val_loss_history.append(float(loss_validate))

        self.best_epoch = (validate_error.index(min(validate_error)) + 1) * 500
        return self.Net, self.best_epoch

    def generate_meta_data(self, X):
        """Generate meta-data for PDE discovery.
        
        Computes various derivatives needed for equation discovery.
        
        Args:
            X: Input data.
            
        Returns:
            tuple: (Theta matrix, cache)
        """
        X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X.requires_grad_(True)

        self.Net.load_state_dict(
            torch.load(f"model_save/Net_{self.best_epoch}.pkl", weights_only=True)
        )
        self.Net.eval()
        u = self.Net(X)
        u_grad = torch.autograd.grad(outputs=u.sum(), inputs=X, create_graph=True)[0]
        ux = u_grad[:, 0].reshape(-1, 1)
        ut = u_grad[:, 1].reshape(-1, 1)
        uxx = torch.autograd.grad(outputs=ux.sum(), inputs=X, create_graph=True)[0][:, 0].reshape(-1, 1)
        uxxx = torch.autograd.grad(outputs=uxx.sum(), inputs=X, create_graph=True)[0][:, 0].reshape(-1, 1)
        utt = torch.autograd.grad(outputs=ut.sum(), inputs=X, create_graph=True)[0][:, 1].reshape(-1, 1)
        Theta = torch.concatenate([u, ux, uxx, uxxx, ut, utt], axis=1)
        self.Theta = Theta

        self.u_t = ut.cpu().detach().numpy()
        self.u_tt = utt.cpu().detach().numpy()

        self.metadata = {
            'u': u.cpu().detach().numpy(),
            'u_x': ux.cpu().detach().numpy(),
            'u_xxx': uxxx.cpu().detach().numpy(),
            'u_t': ut.cpu().detach().numpy()
        }

        return self.Theta

    def random_module(self):
        """Generate a random gene module.
        
        Returns:
            list: Random gene module.
        """
        genes_module = []
        for i in range(self.max_length):
            a = random.randint(0, 3)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module

    def random_genome(self):
        """Generate a random genome (collection of gene modules).
        
        Returns:
            list: Random genome.
        """
        genes = []
        for i in range(self.max_length):
            gene_random = DLGA.random_module(self)
            genes.append(sorted(gene_random))
            prob = random.uniform(0, 1)
            if prob > self.genes_prob:
                break
        return genes

    def translate_DNA(self, gene):
        """Translate gene to mathematical expression.
        
        Args:
            gene: Gene to translate.
            
        Returns:
            tuple: (translated expression, length penalty)
        """
        total = self.Theta.shape[0]
        gene_translate = np.ones([total, 1])
        length_penalty_coef = 0
        for k in range(len(gene)):
            gene_module = gene[k]
            length_penalty_coef += len(gene_module)
            module_out = np.ones([total, 1])
            for i in gene_module:
                temp = self.Theta[:, i].detach().reshape(-1, 1).cpu().numpy()
                module_out *= temp
            gene_translate = np.hstack((gene_translate, module_out))
        gene_translate = np.delete(gene_translate, [0], axis=1)
        return gene_translate, length_penalty_coef

    def get_fitness(self, gene_translate, length_penalty_coef):
        """Calculate fitness of a gene.
        
        Args:
            gene_translate: Translated gene expression.
            length_penalty_coef: Length penalty coefficient.
            
        Returns:
            tuple: (coefficients, MSE, true MSE, equation type)
        """
        # First order derivative fitness
        u_t = self.u_t
        u, d, v = np.linalg.svd(np.hstack((u_t, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, 1)
        res = u_t - np.dot(gene_translate, coef)

        # Second order derivative fitness
        u_tt = self.u_tt
        u, d, v = np.linalg.svd(np.hstack((u_tt, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef_tt = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, 1)
        res_tt = u_tt - np.dot(gene_translate, coef_tt)

        total = self.Theta.shape[0]
        MSE_true = np.sum(np.array(res) ** 2) / total
        MSE_true_tt = np.sum(np.array(res_tt) ** 2) / total

        # Choose better fit between first and second order
        if MSE_true < MSE_true_tt:
            name = "u_t"
            MSE = MSE_true + self.epi * length_penalty_coef
            return coef, MSE, MSE_true, name
        else:
            name = "u_tt"
            MSE = MSE_true_tt + self.epi * length_penalty_coef
            return coef_tt, MSE, MSE_true_tt, name

    def cross_over(self):
        """Perform crossover operation in genetic algorithm.
        
        Returns:
            list: New population after crossover.
        """
        Chrom, size_pop = self.Chrom, self.n_generations
        Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
        for i in range(int(size_pop / 2)):
            n1 = np.random.randint(0, len(Chrom1[i]))
            n2 = np.random.randint(0, len(Chrom2[i]))

            father = Chrom1[i][n1].copy()
            mother = Chrom2[i][n2].copy()

            Chrom1[i][n1] = mother
            Chrom2[i][n2] = father

        Chrom[::2], Chrom[1::2] = Chrom1, Chrom2
        self.Chrom = Chrom
        return self.Chrom

    def mutation(self):
        """Perform mutation operation in genetic algorithm.
        
        Returns:
            list: New population after mutation.
        """
        Chrom, size_pop = self.Chrom, self.pop_size

        for i in range(size_pop):
            # Add module
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom = DLGA.random_module(self)
                if add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)

            # Delete module
            prob = np.random.uniform(0, 1)
            if prob < self.delete_rate:
                if len(Chrom[i]) > 1:
                    delete_index = np.random.randint(0, len(Chrom[i]))
                    Chrom[i].pop(delete_index)

            # Gene mutation
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                if len(Chrom[i]) > 0:
                    n1 = np.random.randint(0, len(Chrom[i]))
                    n2 = np.random.randint(0, len(Chrom[i][n1]))
                    Chrom[i][n1][n2] = random.randint(0, 3)
        self.Chrom = Chrom
        return self.Chrom

    def select(self):
        """Perform selection operation in genetic algorithm."""
        Chrom, size_pop = self.Chrom, self.pop_size
        new_Chrom = []
        new_fitness = []
        new_coef = []
        new_name = []

        fitness_list = []
        coef_list = []
        name_list = []

        # Calculate fitness for all chromosomes
        for i in range(size_pop):
            gene_translate, length_penalty_coef = DLGA.translate_DNA(self, Chrom[i])
            coef, MSE, MSE_true, name = DLGA.get_fitness(
                self, gene_translate, length_penalty_coef
            )
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        # Select best 50% individuals
        re1 = list(
            map(fitness_list.index, heapq.nsmallest(int(size_pop / 2), fitness_list))
        )

        for index in re1:
            new_Chrom.append(Chrom[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        # Fill the rest with new random individuals
        for _ in range(int(size_pop / 2)):
            new = DLGA.random_genome(self)
            new_Chrom.append(new)

        self.Chrom = new_Chrom
        self.Fitness = new_fitness
        self.coef = new_coef
        self.name = new_name

        # Record history (optional)
        if hasattr(self, 'fitness_history'):
            self.fitness_history.append(min(self.Fitness))
        if hasattr(self, 'evolution_history'):
            total_genes = sum(len(chrom) for chrom in self.Chrom)
            unique_genes = len(set(tuple(sorted(module)) 
                                 for chrom in self.Chrom 
                                 for module in chrom))
            self.evolution_history.append({
                'generation': len(self.fitness_history),
                'fitness': min(self.Fitness),
                'complexity': len(self.Chrom[0]),
                'population_size': self.pop_size,
                'unique_modules': unique_genes
            })

        return self.Chrom, self.Fitness, self.coef, self.name

    def delete_duplicates(self):
        """Remove duplicate gene modules from chromosomes.
        
        Returns:
            list: Population with duplicates removed.
        """
        Chrom, size_pop = self.Chrom, self.pop_size
        for i in range(size_pop):
            new_genome = []
            for j in range(len(Chrom[i])):
                if sorted(Chrom[i][j]) not in new_genome:
                    new_genome.append(sorted(Chrom[i][j]))
            Chrom[i] = new_genome
        self.Chrom = Chrom
        return self.Chrom

    def convert_chrom_to_eq(self, chrom, left_name, coef):
        """Convert chromosome to equation string."""
        name = ["u", "ux", "uxx", "uxxx", "ut", "utt"]

        # Debug信息
        print("\nDebug convert_chrom_to_eq:")
        print(f"Chromosome length: {len(chrom)}")
        print(f"Coefficient shape: {coef.shape}")
        print(f"Chromosome structure: {chrom}")
        print(f"Coefficients: {coef}")

        string = []
        try:
            for i in range(len(chrom)):
                if i >= coef.shape[0]:
                    print(f"Warning: More chromosome modules than coefficients at index {i}")
                    break

                item = chrom[i]
                coef_str = str(np.round(coef[i, 0], 4))
                string.append(coef_str)
                string.append("*")

                if not item:
                    print(f"Warning: Empty module at index {i}")
                    continue

                for gene in item:
                    if gene < 0 or gene >= len(name):
                        print(f"Warning: Invalid gene index {gene} at module {i}")
                        continue
                    string.append(name[gene])
                    string.append("*")
                string.pop(-1)
                string.append("+")
                
            if string:
                string.pop(-1)
            
            equation = f"{left_name}=" + "".join(string)
            print(f"Generated equation: {equation}")
            return equation

        except Exception as e:
            print(f"Error in convert_chrom_to_eq: {str(e)}")
            print(f"Current string state: {string}")
            raise

    def evolution(self):
        """Run genetic algorithm evolution following old strategy."""
        self.Chrom = []
        self.Fitness = []
        # Initialize population
        for iter in range(self.pop_size):
            intial_genome = DLGA.random_genome(self)
            self.Chrom.append(intial_genome)
            gene_translate, length_penalty_coef = DLGA.translate_DNA(self, intial_genome)
            coef, MSE, MSE_true, name = DLGA.get_fitness(self, gene_translate, length_penalty_coef)
            self.Fitness.append(MSE)
        DLGA.delete_duplicates(self)

        # Create results directory
        try:
            os.makedirs("result_save/")
        except OSError:
            pass

        # Save parameters
        with open("result_save/DLGA_output.txt", "a") as f:
            f.write("============Params=============\n")
            f.write(f"#l0_penalty:{self.epi}\n")
            f.write(f"#pop_size:{self.pop_size}\n")
            f.write(f"#generations:{self.n_generations}\n")
            f.write("============results=============\n")

        # Evolution loop
        for iter in tqdm(range(self.n_generations)):
            # Save current best individual
            pickle.dump(self.Chrom.copy()[0], open("result_save/best_save.pkl", "wb"))
            best = self.Chrom.copy()[0]
            
            # Genetic operations
            DLGA.cross_over(self)
            DLGA.mutation(self)
            DLGA.delete_duplicates(self)
            
            # Ensure elitism
            best = pickle.load(open("result_save/best_save.pkl", "rb"))
            self.Chrom[0] = best
            
            # Selection and logging
            DLGA.select(self)
            if self.Chrom[0] != best:
                with open("result_save/DLGA_output.txt", "a") as f:
                    f.write(f"iter: {iter + 1}\n")
                    f.write(f"The best Chrom: {self.Chrom[0]}\n")
                    f.write(f"The best coef:  \n{self.coef[0]}\n")
                    f.write(f"The best fitness: {self.Fitness[0]}\n")
                    f.write(f"The best name: {self.name[0]}\n")
                    f.write("----------------------------------------\n")
                print(f"iter: {iter + 1}")
                print(f"The best Chrom: {self.Chrom[0]}")
                print(f"The best coef:  \n{self.coef[0]}")
                print(f"The best fitness: {self.Fitness[0]}")
                print(f"The best name: {self.name[0]}")

        # Final solution
        print("-------------------------------------------")
        print("Finally discovered equation")
        print(f"The best Chrom: {self.Chrom[0]}")
        print(f"The best coef:  \n{self.coef[0]}")
        print(f"The best fitness: {self.Fitness[0]}")
        print(f"The best name: {self.name[0]}")
        print("---------------------------------------------")

        return self.Chrom[0], self.coef[0], self.Fitness[0], self.name[0]

    def fit(self, X, y):
        """Fit model to data.
        
        Args:
            X: Input features.
            y: Target values.
        """
        self.Net, self.best_epoch = self.train_NN(X, y)
        self.Theta = self.generate_meta_data(X)
        
        try:
            Chrom, coef, fitness, name = self.evolution()
            print("\nFinal solution debug info:")
            print(f"Chromosome length: {len(Chrom)}")
            print(f"Coefficient shape: {coef.shape}")
            print(f"Chromosome: {Chrom}")
            print(f"Coefficients: {coef}")
            
            equation = self.convert_chrom_to_eq(Chrom, name, coef)
            print(f"equation form: {equation}")


            if equation_renderer: # 检查导入是否成功
                self.eq_latex = equation_renderer.dlga_eq2latex(
                    chromosome=Chrom,
                    coefficients=coef,
                    lhs_name_str=name
                )
            
        except Exception as e:
            print(f"Error in fit: {str(e)}")
            raise

    def predict(self):
        """Make predictions (not implemented)."""
        pass