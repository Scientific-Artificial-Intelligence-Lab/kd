# kd/model/kd_dlga.py (最终完善版)
import numpy as np
import torch
import random
import os
import pickle
import heapq
from tqdm import tqdm
from kd.model.dlga import DLGA

class KD_DLGA(DLGA):
    """
    DLGA 模型的一个用户友好封装版本。
    """

    def __init__(self, operators: list[str], epi: float, input_dim: int, verbose: bool = True, **kwargs):
        self.user_operators = operators
        self.library_size = len(operators)
        self.verbose = verbose
        super().__init__(epi=epi, input_dim=input_dim, **kwargs)
        self.equations_ = None
        self.best_equation_ = None


    def generate_meta_data(self, X):
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X_tensor.requires_grad_(True)
        self.Net.load_state_dict(
            torch.load(f"model_save/Net_{self.best_epoch}.pkl", map_location=self.device, weights_only=True)
        )
        self.Net.eval()
        u = self.Net(X_tensor)
        u_grad = torch.autograd.grad(outputs=u.sum(), inputs=X_tensor, create_graph=True)[0]
        ux = u_grad[:, 0].reshape(-1, 1) if u_grad.shape[1] > 0 else torch.zeros_like(u)
        ut = u_grad[:, 1].reshape(-1, 1) if u_grad.shape[1] > 1 else torch.zeros_like(u)
        try:
            uxx_grad = torch.autograd.grad(outputs=ux.sum(), inputs=X_tensor, create_graph=True)[0]
            uxx = uxx_grad[:, 0].reshape(-1, 1)
            try:
                uxxx_grad = torch.autograd.grad(outputs=uxx.sum(), inputs=X_tensor, create_graph=True)[0]
                uxxx = uxxx_grad[:, 0].reshape(-1, 1)
            except Exception:
                uxxx = torch.zeros_like(u)
        except Exception:
            uxx = torch.zeros_like(u)
            uxxx = torch.zeros_like(u)
        try:
            utt_grad = torch.autograd.grad(outputs=ut.sum(), inputs=X_tensor, create_graph=True)[0]
            utt = utt_grad[:, 1].reshape(-1, 1)
        except Exception:
            utt = torch.zeros_like(u)
        available_ops = {'u': u, 'u_x': ux, 'u_t': ut, 'u_xx': uxx, 'u_xxx': uxxx, 'u_tt': utt}
        # 1. Theta 矩阵的构建依然严格遵守 user_operators，这用于方程搜索
        theta_columns = []
        for op_name in self.user_operators:
            if op_name in available_ops:
                theta_columns.append(available_ops[op_name])
        self.Theta = torch.concatenate(theta_columns, axis=1) if theta_columns else torch.empty(X.shape[0], 0).to(self.device)

        # 2. metadata 则存储所有计算出的项，使其成为一个完整的“数据源”
        self.metadata = {key: val.cpu().detach().numpy() for key, val in available_ops.items()}
        
        # 3. 父类中其他需要的属性也需更新
        self.u_t = ut.cpu().detach().numpy()
        self.u_tt = utt.cpu().detach().numpy()

        return self.Theta

    def random_module(self):
        genes_module = []
        for _ in range(self.max_length):
            if self.library_size == 0: break
            a = random.randint(0, self.library_size - 1)
            genes_module.append(a)
            if random.uniform(0, 1) > self.partial_prob: break
        return genes_module

    def mutation(self):
        Chrom, size_pop = self.Chrom, self.pop_size
        for i in range(size_pop):
            if random.uniform(0, 1) < self.add_rate:
                add_Chrom = self.random_module()
                if add_Chrom and add_Chrom not in Chrom[i]: Chrom[i].append(add_Chrom)
            if random.uniform(0, 1) < self.delete_rate and len(Chrom[i]) > 1:
                Chrom[i].pop(random.randint(0, len(Chrom[i]) - 1))
            if random.uniform(0, 1) < self.mutate_rate and self.library_size > 0:
                if Chrom[i]:
                    try:
                        n1 = random.randint(0, len(Chrom[i]) - 1)
                        if Chrom[i][n1]:
                            n2 = random.randint(0, len(Chrom[i][n1]) - 1)
                            Chrom[i][n1][n2] = random.randint(0, self.library_size - 1)
                    except ValueError:
                        continue
        self.Chrom = Chrom
        return self.Chrom

    def random_genome(self):
        genes = []
        for _ in range(self.max_length):
            gene_random = self.random_module()
            if gene_random: genes.append(sorted(gene_random))
            if random.uniform(0, 1) > self.genes_prob: break
        return genes

    def convert_chrom_to_eq(self, chrom, left_name, coef):
        name = self.user_operators
        string = []
        if not chrom: return f"{left_name}= (empty equation)"
        for i in range(len(chrom)):
            if i >= len(coef): break
            item = chrom[i]
            coef_str = str(np.round(coef[i, 0], 4))
            term_str = "*".join([name[gene] for gene in item if gene < len(name)])
            if not term_str: continue
            string.append(f"{coef_str}*{term_str}")
        equation = f"{left_name}=" + "+".join(string).replace("+-", "-")
        return equation

    def evolution(self):
        self.Chrom = []
        self.Fitness = []
        for _ in range(self.pop_size):
            initial_genome = self.random_genome()
            self.Chrom.append(initial_genome)
            if not initial_genome:
                self.Fitness.append(float('inf'))
                continue
            gene_translate, length_penalty_coef = self.translate_DNA(initial_genome)
            _, MSE, _, _ = self.get_fitness(gene_translate, length_penalty_coef)
            self.Fitness.append(MSE)
        self.delete_duplicates()
        os.makedirs("result_save/", exist_ok=True)

        for iter_num in tqdm(range(self.n_generations)):
            best_chrom_path = "result_save/best_save.pkl"
            best = []
            if self.Chrom:
                best = self.Chrom[0]
                with open(best_chrom_path, "wb") as f:
                    pickle.dump(best, f)
            self.cross_over()
            self.mutation()
            self.delete_duplicates()
            if os.path.exists(best_chrom_path):
                with open(best_chrom_path, "rb") as f:
                    best_loaded = pickle.load(f)
                if best_loaded:
                    if self.Chrom:
                        self.Chrom[0] = best_loaded
                    else:
                        self.Chrom.append(best_loaded)
            self.select()
            if self.verbose and self.Chrom and self.Chrom[0] != best:
                output_str = (f"iter: {iter_num + 1}\n"
                              f"The best Chrom: {self.Chrom[0]}\n"
                              f"The best coef:  \n{self.coef[0]}\n"
                              f"The best fitness: {self.Fitness[0]}\n"
                              f"The best name: {self.name[0]}\n"
                              "----------------------------------------\n")
                with open("result_save/DLGA_output.txt", "a") as f:
                    f.write(output_str)
                print(output_str)
        if not self.Chrom:
            print("Evolution finished, but no solution was found.")
            return [], [], float('inf'), ""
        return self.Chrom[0], self.coef[0], self.Fitness[0], self.name[0]

    def select(self):
        # ... (选择逻辑与上一版本相同) ...
        Chrom, size_pop = self.Chrom, self.pop_size
        new_Chrom, new_fitness, new_coef, new_name = [], [], [], []
        fitness_list, coef_list, name_list = [], [], []
        for i in range(size_pop):
            if not Chrom[i]:
                fitness_list.append(float('inf'))
                coef_list.append([])
                name_list.append("")
                continue
            gene_translate, length_penalty_coef = self.translate_DNA(Chrom[i])
            coef, MSE, _, name = self.get_fitness(gene_translate, length_penalty_coef)
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        num_to_keep = int(size_pop / 2)
        if len(fitness_list) < num_to_keep:
            num_to_keep = len(fitness_list)
        re1 = list(map(fitness_list.index, heapq.nsmallest(num_to_keep, fitness_list)))
        for index in re1:
            new_Chrom.append(Chrom[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        for _ in range(size_pop - len(new_Chrom)):
            new = self.random_genome()
            new_Chrom.append(new)
        self.Chrom = new_Chrom
        self.Fitness = new_fitness
        self.coef = new_coef
        self.name = new_name
        
        # 确保 Fitness 和 Chrom 列表不为空，以避免 min() 或索引错误
        if self.Fitness and self.Chrom:
            # 记录最佳适应度
            if hasattr(self, 'fitness_history'):
                self.fitness_history.append(min(self.Fitness))
            
            # 记录详细的进化历史数据，供后续可视化使用
            if hasattr(self, 'evolution_history'):
                unique_genes = len(set(tuple(sorted(module)) 
                                     for chrom in self.Chrom if chrom
                                     for module in chrom))
                self.evolution_history.append({
                    'generation': len(self.fitness_history),
                    'fitness': min(self.Fitness),
                    'complexity': len(self.Chrom[0]) if self.Chrom[0] else 0,
                    'population_size': self.pop_size,
                    'unique_modules': unique_genes
                })

        return self.Chrom, self.Fitness, self.coef, self.name
    
    def fit(self, X, y):
        """
        重写 fit 方法，以确保在调用渲染器时传递正确的动态算子列表
        """
        self.Net, self.best_epoch = self.train_NN(X, y)
        self.Theta = self.generate_meta_data(X)
        
        try:
            # 调用我们重写过的 evolution 方法
            Chrom, coef, fitness, name = self.evolution()
            
            print("\nFinal solution debug info:")
            print(f"Chromosome length: {len(Chrom)}")
            print(f"Coefficient shape: {coef.shape}")
            print(f"Chromosome: {Chrom}")
            print(f"Coefficients: {coef}")
            
            # 调用我们重写过的 convert_chrom_to_eq 方法
            equation = self.convert_chrom_to_eq(Chrom, name, coef)
            print(f"equation form: {equation}")

            # 检查 equation_renderer 是否可用
            try:
                from kd.viz import dlga_eq2latex as equation_renderer
                # 【关键修改】 调用新的渲染器，并传入 operator_names 参数
                self.eq_latex = equation_renderer.dlga_eq2latex(
                    chromosome=Chrom,
                    coefficients=coef,
                    lhs_name_str=name,
                    operator_names=self.user_operators # 将我们的动态列表传进去
                )
            except ImportError:
                print("\n[INFO] Equation renderer not found or failed to import. Skipping LaTeX generation.")
            
        except Exception as e:
            print(f"Error in fit: {str(e)}")
            # 在出错时也打印有用的调试信息
            import traceback
            traceback.print_exc()
            raise
        
    def predict(self, mesh_data):
        X_tensor = torch.from_numpy(mesh_data.astype(np.float32)).to(self.device) # Convert to tensor and predict
        with torch.no_grad():
            u_pred = self.Net(X_tensor).cpu().numpy()
        return u_pred