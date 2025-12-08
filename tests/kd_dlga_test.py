import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
import os
from kd.model.kd_dlga import KD_DLGA
from kd.dataset import PDEDataset

# The original DLGA class depends on an NN class. For testing purposes,
# we need a placeholder that mimics the original structure.
# We can define a minimal mock NN class right inside the test file.
class MockNN(torch.nn.Module):
    def __init__(self, Input_Dim, Num_Hidden_Layers, Neurons_Per_Layer, Output_Dim, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(Input_Dim, Neurons_Per_Layer))
        for _ in range(Num_Hidden_Layers):
            self.layers.append(torch.nn.Linear(Neurons_Per_Layer, Neurons_Per_Layer))
        self.layers.append(torch.nn.Linear(Neurons_Per_Layer, Output_Dim))

    def forward(self, x):
        for layer in self.layers:
            x = torch.sin(layer(x)) # Using sin as in the original
        return x

def test_initialization_succeeds():
    """
    Tests that the KD_DLGA class can be initialized correctly.
    This test should already be passing from your colleague's work.
    """
    operators = ['u', 'u_x']
    model = KD_DLGA(operators=operators, epi=0.01, input_dim=2)
    assert model.user_operators == operators
    assert model.epi == 0.01

# === OUR NEW TEST ===
def test_generate_meta_data_override(tmp_path):
    """
    Tests that the overridden generate_meta_data method correctly
    builds the Theta matrix based on the user_operators list.
    """
    # 1. Define custom operators and create model instance
    custom_operators = ['u', 'u_x', 'u_xx']
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2, max_iter=100)

    # 2. The method requires a saved model file. We'll create a dummy one.
    #    - Replace the model's Net with our MockNN for this test
    model.Net = MockNN(Input_Dim=2, Num_Hidden_Layers=1, Neurons_Per_Layer=10, Output_Dim=1)
    #    - Create a dummy model directory
    model_save_dir = tmp_path / "model_save"
    model_save_dir.mkdir()
    #    - Set the best_epoch and save the dummy network state
    model.best_epoch = 500 # A dummy epoch number
    dummy_model_path = model_save_dir / f"Net_{model.best_epoch}.pkl"
    torch.save(model.Net.state_dict(), dummy_model_path)
    
    # 3. Create dummy input data
    X_test = np.random.rand(10, 2) # 10 data points, 2 features (e.g., x and t)

    # 4. We need to call generate_meta_data. It expects to be called from a
    #    directory where 'model_save/...' is accessible. We'll temporarily
    #    change the working directory.
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Call the method we want to test
        model.generate_meta_data(X_test)
        
        # 5. Assert that the generated Theta matrix has the correct number of columns
        assert model.Theta is not None
        assert model.Theta.shape[1] == len(custom_operators)
        
    finally:
        # Change back to the original directory
        os.chdir(original_cwd)

def test_random_genome_respects_operator_bounds():
    """
    测试: 随机生成的基因组(genome)中的所有基因，其索引值
    必须小于自定义算子列表的长度。
    """
    # 1. 使用一个很小的算子列表，其长度为 2 (合法索引为 0, 1)
    custom_operators = ['u', 'u_x']
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # 2. 调用父类的 random_genome 方法生成一个随机基因组
    # 这个方法内部会调用 random_module 方法
    genome = model.random_genome()

    # 3. 断言：检查基因组中的每一个基因值，确保它小于 2
    # 如果出现 >= 2 的值，说明原始的 hardcoded `random.randint(0, 3)` 被调用了
    assert genome, "Genome should not be empty"
    for module in genome:
        for gene_index in module:
            assert gene_index < len(custom_operators)

def test_mutation_respects_operator_bounds():
    """
    测试: 基因变异过程所产生的新的基因，其索引值也
    必须小于自定义算子列表的长度。
    """
    # 1. 同样，使用一个长度为 2 的小算子列表
    custom_operators = ['u', 'u_x']
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # 2. 我们手动创建一个合法的种群 Chrom，并强制变异率
    #    为 100%，以确保变异必定发生。
    model.mutate_rate = 1.0
    model.pop_size = 1
    model.Chrom = [ [[0], [1]] ] # 一个完全合法的染色体 [[u], [u_x]]

    # 3. 运行变异方法
    model.mutation()

    # 4. 断言：检查变异后的染色体，确保所有基因值依然小于 2
    mutated_genome = model.Chrom[0]
    assert mutated_genome, "Mutated genome should not be empty"
    for module in mutated_genome:
        for gene_index in module:
            assert gene_index < len(custom_operators)


def test_convert_chrom_to_eq_uses_custom_operators():
    """
    测试: 将染色体翻译成方程字符串时，必须使用自定义的算子名称。
    """
    # 1. 使用一个非标准顺序的算子列表，以验证映射关系
    custom_operators = ['u_xx', 'u', 'u_t']
    model = KD_DLGA(operators=custom_operators, epi=0.01, input_dim=2)

    # 2. 手动创建一个染色体和对应的系数
    # 代表方程: u_t = 2.5 * u_xx + -1.0 * u
    best_chrom = [[0], [1]] # 对应 ['u_xx', 'u']
    best_coef = np.array([[2.5], [-1.0]])
    left_hand_side = 'u_t'

    # 3. 调用我们想要测试的方法
    equation_str = model.convert_chrom_to_eq(best_chrom, left_hand_side, best_coef)

    # 4. 断言：生成的字符串应该包含正确的项
    #    原始方法会错误地生成 "2.5*u + -1.0*ux"
    assert "2.5*u_xx" in equation_str
    assert "-1.0*u" in equation_str
    assert "u_t=" in equation_str


def _make_toy_pdedataset():
    x = np.linspace(0.0, 1.0, 4)
    t = np.linspace(0.0, 1.0, 5)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    usol = np.sin(xx) * np.cos(tt)
    return PDEDataset(
        equation_name="toy",
        pde_data=None,
        domain=None,
        epi=0.0,
        x=x,
        t=t,
        usol=usol,
    )


def test_fit_dataset_uses_sample_and_calls_fit(monkeypatch):
    dataset = _make_toy_pdedataset()
    model = KD_DLGA(operators=["u", "u_x"], epi=0.01, input_dim=2, max_iter=10)

    called = {}

    def fake_fit(self, X, y):
        called["X"] = X
        called["y"] = y
        return self

    monkeypatch.setattr(KD_DLGA, "fit", fake_fit, raising=False)

    result = model.fit_dataset(dataset, sample=5, sample_method="random")

    assert result is model
    assert model.dataset_ is dataset
    assert "X" in called and "y" in called
    assert called["X"].shape[0] == 5
    assert called["X"].shape[1] == 2
    assert called["y"].shape[0] == 5


def test_fit_dataset_sample_none_uses_full_mesh(monkeypatch):
    dataset = _make_toy_pdedataset()
    total_points = len(dataset.x) * len(dataset.t)

    model = KD_DLGA(operators=["u"], epi=0.01, input_dim=2, max_iter=10)

    called = {}

    def fake_fit(self, X, y):
        called["X"] = X
        called["y"] = y
        return self

    monkeypatch.setattr(KD_DLGA, "fit", fake_fit, raising=False)

    result = model.fit_dataset(dataset, sample=None)

    assert result is model
    assert model.dataset_ is dataset
    assert called["X"].shape[0] == total_points
    assert called["X"].shape[1] == 2
    assert called["y"].shape[0] == total_points


def test_fit_dataset_requires_pdedataset():
    model = KD_DLGA(operators=["u"], epi=0.01, input_dim=2, max_iter=10)
    with pytest.raises(TypeError):
        model.fit_dataset("not-a-dataset")  # type: ignore[arg-type]
