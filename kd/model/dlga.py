from abc import ABCMeta, abstractmethod

import torch.utils.data
from ..base import BaseEstimator
from ..utils.utils_GA import *
import numpy as np
import heapq
from tqdm import tqdm
import pickle

from kd.vizr.vizr import *


class BaseGa(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit model"""

    def predict(self, X):
        pass


class DLGA(BaseGa):
    _parameter: dict = {}

    def __init__(self, epi, input_dim):
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

        self.epi = epi
        self.fitness_history = []  # 用来track每一代的best fitness
        self.vizr = Vizr("DLGA Training Progress", nrows=1, ncols=2)

        # Add plots for training and validation loss
        self.vizr.add(LinePlot, "train_loss", id=0, color="red")
        self.vizr.add(LinePlot, "valid_loss", id=1, color="blue")

        # Set labels for subplot
        self.vizr.axes[0].set_xlabel("Iteration")
        self.vizr.axes[0].set_ylabel("train_loss")

        self.vizr.axes[1].set_xlabel("Iteration")
        self.vizr.axes[1].set_ylabel("valid_loss")

    def train_NN(self, X, y):
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)
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
        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        X_valid = torch.from_numpy(X_valid.astype(np.float32)).to(self.device)

        NN_optimizer = torch.optim.Adam(
            [
                {"params": self.Net.parameters()},
            ]
        )

        MSELoss = torch.nn.MSELoss()
        validate_error = []
        try:
            os.makedirs(f"model_save/")
        except OSError:
            pass
        print(f"===============train Net=================")

        for iter in range(50000):
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

                self.vizr.update("train_loss", iter + 1, float(loss), id=0).update(
                    "valid_loss", iter + 1, float(loss_validate), id=1
                ).render()

        self.best_epoch = (validate_error.index(min(validate_error)) + 1) * 500
        return self.Net, self.best_epoch

    def generate_meta_data(self, X):
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
        uxx = torch.autograd.grad(outputs=ux.sum(), inputs=X, create_graph=True)[0][
            :, 0
        ].reshape(-1, 1)
        uxxx = torch.autograd.grad(outputs=uxx.sum(), inputs=X, create_graph=True)[0][
            :, 0
        ].reshape(-1, 1)
        utt = torch.autograd.grad(outputs=ut.sum(), inputs=X, create_graph=True)[0][
            :, 1
        ].reshape(-1, 1)
        Theta = torch.concatenate([u, ux, uxx, uxxx, ut, utt], axis=1)
        self.Theta = Theta

        self.u_t = ut.cpu().detach().numpy()
        self.u_tt = utt.cpu().detach().numpy()

        return self.Theta

    def random_module(self):
        genes_module = []
        for i in range(self.max_length):
            a = random.randint(0, 3)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module

    def random_genome(self):
        genes = []

        for i in range(self.max_length):
            gene_random = DLGA.random_module(self)
            genes.append(sorted(gene_random))
            prob = random.uniform(0, 1)
            if prob > self.genes_prob:
                break
        return genes

    def translate_DNA(self, gene):
        total = self.Theta.shape[0]
        gene_translate = np.ones([total, 1])
        length_penalty_coef = 0
        for k in range(len(gene)):
            gene_module = gene[k]
            length_penalty_coef += len(gene_module)
            module_out = np.ones([self.Theta.shape[0], 1])
            for i in gene_module:
                temp = self.Theta[:, i].detach().reshape(-1, 1).cpu().numpy()
                module_out *= temp
            gene_translate = np.hstack((gene_translate, module_out))
        gene_translate = np.delete(gene_translate, [0], axis=1)
        return gene_translate, length_penalty_coef

    def get_fitness(self, gene_translate, length_penalty_coef):
        u_t = self.u_t
        u, d, v = np.linalg.svd(np.hstack((u_t, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, 1)
        res = u_t - np.dot(gene_translate, coef)

        u_tt = self.u_tt
        u, d, v = np.linalg.svd(np.hstack((u_tt, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef_tt = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, 1)
        res_tt = u_tt - np.dot(gene_translate, coef_tt)

        total = self.Theta.shape[0]
        MSE_true = np.sum(np.array(res) ** 2) / total
        MSE_true_tt = np.sum(np.array(res_tt) ** 2) / total

        if MSE_true < MSE_true_tt:
            name = "u_t"
            MSE = MSE_true + self.epi * length_penalty_coef
            return coef, MSE, MSE_true, name

        else:
            name = "u_tt"
            MSE = MSE_true_tt + self.epi * length_penalty_coef
            return coef_tt, MSE, MSE_true_tt, name

    def cross_over(self):
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
        Chrom, size_pop = self.Chrom, self.pop_size

        for i in range(size_pop):
            n1 = np.random.randint(0, len(Chrom[i]))

            # ------------add module---------------
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom = DLGA.random_module(self)
                if add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)

            # --------delete module----------------
            prob = np.random.uniform(0, 1)
            if prob < self.delete_rate:
                if len(Chrom[i]) > 1:
                    delete_index = np.random.randint(0, len(Chrom[i]))
                    Chrom[i].pop(delete_index)

            # ------------gene mutation------------------
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                if len(Chrom[i]) > 0:
                    n1 = np.random.randint(0, len(Chrom[i]))
                    n2 = np.random.randint(0, len(Chrom[i][n1]))
                    Chrom[i][n1][n2] = random.randint(0, 3)
        self.Chrom = Chrom
        return self.Chrom

    def select(self):  # nature selection wrt pop's fitness
        Chrom, size_pop = self.Chrom, self.pop_size
        new_Chrom = []
        new_fitness = []
        new_coef = []
        new_name = []

        fitness_list = []
        coef_list = []
        name_list = []

        for i in range(size_pop):
            gene_translate, length_penalty_coef = DLGA.translate_DNA(self, Chrom[i])
            coef, MSE, MSE_true, name = DLGA.get_fitness(
                self, gene_translate, length_penalty_coef
            )
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        re1 = list(
            map(fitness_list.index, heapq.nsmallest(int(size_pop / 2), fitness_list))
        )

        for index in re1:
            new_Chrom.append(Chrom[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        for index in range(int(size_pop / 2)):
            new = DLGA.random_genome(self)
            new_Chrom.append(new)

        self.Chrom = new_Chrom
        self.Fitness = new_fitness
        self.coef = new_coef
        self.name = new_name
        return self.Chrom, self.Fitness, self.coef, self.name

    def delete_duplicates(self):
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
        name = ["u", "ux", "uxx", "uxxx", "ut", "utt"]
        string = []
        for i in range(len(chrom)):
            item = chrom[i]
            string.append(str(np.round(coef[i, 0], 4)))
            string.append("*")
            for gene in item:
                string.append(name[gene])
                string.append("*")
            string.pop(-1)
            string.append("+")
        string.pop(-1)
        string = f"{left_name}=" + "".join(string)
        return string

    def evolution(self):
        self.Chrom = []
        self.Fitness = []
        self.fitness_history = []

        # Add GA evolution subplot
        ga_plot_idx = self.vizr.add_subplot()
        self.vizr.add(LinePlot, "best_fitness", id=ga_plot_idx, color="green")

        # Set labels for GA subplot
        self.vizr.axes[ga_plot_idx].set_xlabel("Generation")
        self.vizr.axes[ga_plot_idx].set_ylabel("Best Fitness")
        self.vizr.axes[ga_plot_idx].set_title("GA Evolution")

        # Initial population
        for iter in range(self.pop_size):
            intial_genome = DLGA.random_genome(self)
            self.Chrom.append(intial_genome)
            gene_translate, length_penalty_coef = DLGA.translate_DNA(
                self, intial_genome
            )
            coef, MSE, MSE_true, name = DLGA.get_fitness(
                self, gene_translate, length_penalty_coef
            )
            self.Fitness.append(MSE)

            # Update visualization
            self.vizr.update(
                "best_fitness", iter + 1, float(MSE), id=ga_plot_idx
            ).render()

        DLGA.delete_duplicates(self)
        try:
            os.makedirs(f"result_save/")
        except OSError:
            pass

        with open(f"result_save/DLGA_output.txt", "a") as f:
            f.write(f"============Params=============\n")
            f.write(f"#l0_penalty:{self.epi}\n")
            f.write(f"#pop_size:{self.pop_size}\n")
            f.write(f"#generations:{self.n_generations}\n")
            f.write(f"============results=============\n")
        for iter in tqdm(range(self.n_generations)):
            pickle.dump(self.Chrom.copy()[0], open(f"result_save/best_save.pkl", "wb"))
            best = self.Chrom.copy()[0]
            DLGA.cross_over(self)
            DLGA.mutation(self)
            DLGA.delete_duplicates(self)
            best = pickle.load(open(f"result_save/best_save.pkl", "rb"))
            self.Chrom[0] = best
            DLGA.select(self)
            if self.Chrom[0] != best:
                with open(f"result_save/DLGA_output.txt", "a") as f:
                    f.write(f"iter: {iter + 1}\n")
                    f.write(f"The best Chrom: {self.Chrom[0]}\n")
                    f.write(f"The best coef:  \n{self.coef[0]}\n")
                    f.write(f"The best fitness: {self.Fitness[0]}\n")
                    f.write(f"The best name: {self.name[0]}\n")
                    f.write(f"----------------------------------------\n")
                    print(f"iter: {iter + 1}\n")
                    print(f"The best Chrom: {self.Chrom[0]}")
                    print(f"The best coef:  \n{self.coef[0]}")
                    print(f"The best fitness: {self.Fitness[0]}")
                    print(f"The best name: {self.name[0]}\r")

        print("-------------------------------------------")
        print(f"Finally discovered equation")
        print(f"The best Chrom: {self.Chrom[0]}")
        print(f"The best coef:  \n{self.coef[0]}")
        print(f"The best fitness: {self.Fitness[0]}")
        print(f"The best name: {self.name[0]}\r")
        print("---------------------------------------------")

        return self.Chrom[0], self.coef[0], self.Fitness[0], self.name[0]

    def fit(self, X, y):
        self.Net, self.best_epoch = self.train_NN(X, y)
        self.Theta = self.generate_meta_data(X)
        Chrom, coef, _, name = self.evolution()
        print("equation form:", self.convert_chrom_to_eq(Chrom, name, coef))
        self.vizr.show()

    def predict(self):
        pass
