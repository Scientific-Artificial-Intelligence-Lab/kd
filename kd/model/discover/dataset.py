from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.io as scio
from pyDOE import lhs

from discover.task.pde.utils_fd import FiniteDiff


data_list = [
    'chafee-infante', 'Burgers', 'Kdv'
]

class BaseDataHandler(ABC):
    def __init__(self, source=None):
        self.source = source
        self.data = {}

    @abstractmethod
    def process_data(self):
        pass

    def get_data(self):
        return self.data
    

class RegularData(BaseDataHandler):
    
    def __init__(self, source):
        super().__init__(source)
        
        self.X = []
        self.sym_true= None
        self.n_input_dim = 1 # default 1 spatial dimension
        if isinstance(self.source, str):
            assert self.source in data_list, "Provided data is not included"
              
    def process_data(self):
        super().process_data()
        """
        process regular data
        """
        pass
    
    def load_data(self,):
        folder_name ='./discover/data_file'
        if self.source  == 'chafee-infante': # 301*200

            u = np.load(f"{folder_name}/chafee_infante_CI.npy")
            x = np.load(f"{folder_name}/chafee_infante_x.npy").reshape(-1,1)
            t = np.load(f"{folder_name}/chafee_infante_t.npy").reshape(-1,1)
            self.sym_true = 'add,add,u1,n3,u1,diff2,u1,x1'

        elif self.source  == 'Burgers':

            data = scio.loadmat(f"{folder_name}/burgers.mat")
            u=data.get("usol")
            x=np.squeeze(data.get("x")).reshape(-1,1)
            t=np.squeeze(data.get("t").reshape(-1,1))
            self.sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'

        elif self.source  == 'Kdv':      
            data = scio.loadmat('./discover/task/pde/data_new/Kdv.mat')
            u=data.get("uu")
            n,m=u.shape
            x=np.squeeze(data.get("x")).reshape(-1,1)
            t=np.squeeze(data.get("tt").reshape(-1,1))

            n,m = u.shape #512, 201
            dt = t[1]-t[0]
            dx = x[1]-x[0]
            self.sym_true = 'add,mul,u1,diff,u1,x1,diff3,u1,x1'
        else:
            raise NotImplemented
            
        n, m = u.shape
        self.ut = np.zeros((n, m))
        dt = t[1]-t[0]
        self.X.append(x)
        self.u = u

        for idx in range(n):
            self.ut[idx, :] = FiniteDiff(u[idx, :], dt)
        
        self.data = {
            'u': self.u,
            'ut' : self.ut,
            'X' : self.X,
            'n_input_dim': self.n_input_dim,
            'sym_true': self.sym_true
        }
        
    def info(self):
        print()
            


class SpareseData(BaseDataHandler):
    
    def __init__(self,xt,u,*args, **kwargs):
        super().__init__(*args, **kwargs)    
        
        self.xt = xt
        self.x = xt[:,:-1] # spatial coordinates
        self.t = xt[:,-1:] # time coorinates
        self.u = u # shape = (N, 1)
        
        self.colloc_num = 20000
        
        
    def process_data(self,domains):
        super().process_data()
        
        # 1. train test split
        Split_TrainVal = 0.8
        Num_d = self.u.shape[0]
        N_u_train = int(Num_d*Split_TrainVal)
        idx_train = np.random.choice(Num_d, N_u_train, replace=False)
        self.X_u_train = self.xt[idx_train,:]
        self.u_train = self.u[idx_train,:]
        idx_val = np.setdiff1d(np.arange(Num_d), idx_train, assume_unique=True)
        self.X_u_val = self.xt[idx_val,:]
        self.u_val = self.u[idx_val,:]
        
        #2. Collocation points sampling
        lb,ub = domains[0], domains[1]
        X_f_train = lb + (ub-lb)*lhs(2, self.colloc_num)     
        self.X_f_train = np.vstack((X_f_train, self.X_u_train))   

        self.data = {
            'X_u_train': self.X_u_train,
            'u_train': self.u_train,
            'X_f_train': self.X_f_train,
            'X_u_val': self.X_u_val,
            'u_val': self.u_val,
            'lb':lb,
            'ub':ub
        }
        
    

