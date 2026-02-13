import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.stats import pearsonr


from discover.task import HierarchicalTask
from discover.library import Library
from discover.functions import create_tokens
from discover.task.pde.data_load import *
from discover.task.pde.utils_nn import load_noise_data
from discover.task.pde.utils_noise import *


class PDETask(HierarchicalTask):
    """
    Class for the common PDE disovery task (MODE1). 
    """

    task_type = "pde"

    def __init__(self, function_set, dataset, metric="residual",
                metric_params=(0.01,), extra_metric_test=None,
                extra_metric_test_params=(), reward_noise=0.0,
                reward_noise_type="r", threshold=1e-12,
                data_noise_level=0,
                data_amount = 1,
                use_meta_data = False,
                use_torch = False,
                sym_true_input =None,
                max_depth=4,
                normalize_variance=False, protected=False,
                spatial_error = True, 
                decision_tree_threshold_set=None,
                cut_ratio = 0.03,
                data_info = None,
                add_const = False,
                eq_num=1
                ):
        """
        Parameters
        ----------
        function_set : list or None
            List of allowable functions. If None, uses function_set according to
            benchmark dataset.

        dataset : dict, str, or tuple
            If dict: .dataset.BenchmarkDataset kwargs.
            If str ending with .csv: filename of dataset.
            If other str: name of benchmark dataset.
            If tuple: (X, y) data

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        extra_metric_test : str
            Name of extra function metric to use for testing.

        extra_metric_test_params : list
            List of metric-specific parameters for extra test metric.

        reward_noise : float
            Noise level to use when computing reward.

        reward_noise_type : "y_hat" or "r"
            "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
            "r" : N(0, reward_noise) is added to r.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision trees.
        
        eq_num: int
            The number of governing equations to be identified
        """

        super(HierarchicalTask).__init__()

        self.add_const=add_const
        self.eq_num = eq_num
        # self.X_test = self.y_test = self.y_test_noiseless = None
        self.name = dataset
        self.noise_level = data_noise_level
        self.spatial_error = spatial_error
        self.cut_ratio = cut_ratio
        self.max_depth = max_depth
        
        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.metric, self.invalid_reward, self.max_reward = make_pde_metric(metric, *metric_params)
        self.extra_metric_test = extra_metric_test
        self.metric_test = None

        self.rng = None
        self.scale = None
        self.function_set=function_set
        self.decision_tree_threshold_set =decision_tree_threshold_set
        self.protected = protected

        self.stochastic = reward_noise > 0.0

    def load_data(self, dataset):
        self.u = [dataset['u']]
        self.x = dataset['X']
        self.ut = dataset['ut'].reshape(-1,1)
        if torch.is_tensor(self.ut):
            self.ut = tensor2np(self.ut)
        self.sym_true = dataset.get('sym_true', '')
        self.n_input_var = dataset.get('n_input_dim', 1)
        self.u_test,self.ut_test = dataset.get('u_test',None),dataset.get('ut_test', None)

        # Parameter fields (non-differentiated variables).
        self.param_data = dataset.get('param_data', None)
        self.n_param_var = dataset.get('n_param_var', 0)
        self.param_names = dataset.get('param_names', None)

        # Set the Library
        tokens = create_tokens(n_input_var=self.n_input_var,
                               function_set=self.function_set,
                               protected=self.protected,
                               n_state_var=len(self.u),
                               decision_tree_threshold_set=self.decision_tree_threshold_set,
                               task_type='pde',
                               n_param_var=self.n_param_var,
                               param_names=self.param_names)
        self.library = Library(tokens)

        # Bind param data references onto param tokens so that execute.py
        # can retrieve the data without changing call signatures.
        if self.param_data:
            for token in self.library.tokens:
                if token.param_var is not None and token.param_var < len(self.param_data):
                    token._param_data_ref = self.param_data[token.param_var]
            
        # if '2D' in dataset:
        #     load_class = load_data_2D 
        # elif "para" in dataset:
        #     load_class = load_param_data
        # elif "MD_NU" in dataset: # multi input and multi-dimension
        #     load_class = load_data_MD_NU
        # else:
        #     load_class = load_data    
        
        
    def reward_function(self,p):
        # 需要额外获取 y_right
        y_hat, y_right, w = p.execute(self.u, self.x, self.ut)
        n = len(w)
        if p.invalid:
            # print(p.tokens)
            
            return self.invalid_reward, [0], None, None # 也注意修改
        
        r = self.metric(self.ut, y_hat,n)

        return r, w, y_hat, y_right # 多拿俩返回值


    def mse_function(self,p):
        """
        task function utilized to calculate mse for coeffcients
        """
        y_hat, y_right, w = p.execute(self.u, self.x, self.ut)
        diffs = y_hat-self.ut
        loss = (np.mean(np.square(diffs))) 
        return loss


    def evaluate(self, p):
        # Compute predictions on test data
        y_hat,y_right,  w = p.execute(self.u, self.x, self.ut)

        n = len(w)
        # y_hat = p.execute(self.X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((self.ut - y_hat) ** 2)

            # NMSE on noiseless test data (used to determine recovery)
            # nmse_test_noiseless = np.mean((self.y_test_noiseless - y_hat) ** 2) / self.var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test < self.threshold

        info = {
            "nmse_test" : nmse_test,
            "success" : success,

        }
        if self.u_test is not None:
        
            y_hat_test,y_right, w_test = p.execute(self.u_test, self.x, self.ut_test, test=True)
            info.update({
                'w_test': w_test
            })

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat,n)
                m_test_noiseless = self.metric_test(self.y_test_noiseless, y_hat,n)

            info.update({
                self.extra_metric_test : m_test,
                self.extra_metric_test + '_noiseless' : m_test_noiseless
            })

        return info
    
    def evaluate_diff(self, p):
        y_hat,y_right,  w = p.execute(self.u, self.x, self.ut)
        return self.ut-y_hat
        
    def set_ut(self, ut_diff):
        self.ut=ut_diff

    def reset_ut(self,id ):
        assert self.ut_cache is not None
        self.ut = self.ut_cache[id]


    def terms_values(self, p):
        '''return results list and terms list
            results:  shape = [(-1)]
            terms: tokens
        '''
        values = p.execute_terms(self.u, self.x)
        tokens = p.STRidge.terms_token
        # list
        return tokens, values


def make_pde_metric(name, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """
    # ut = 
    all_metrics = {

        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + np.sqrt(np.mean((y - y_hat)**2)/np.var(y))),
                        1),

        "pde_reward":  (lambda y, y_hat,n : (1-args[0]*n)/(1 + np.sqrt(np.mean((y - y_hat)**2)/np.var(y))),
                        1),

        "correlation": (lambda y,y_hat,n: pearsonr(y.ravel(), y_hat.ravel())[0],
                        1),
        "R2":(lambda y, y_hat,n: 1 - ((y_hat - y)**2).sum() / ((y - y.mean())**2).sum() )

      
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
       
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        'pde_reward':0.0,
        'correlation':0.0

    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "inv_nrmse" : 1.0,
        "pde_reward":1.0,
        'correlation':1.0

    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward
