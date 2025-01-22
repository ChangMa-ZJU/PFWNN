# /*
#  * @Author: chang.ma 
#  * @Date: 2023-12-09 14:32:32
#  * @Last Modified by: chang.ma 
#  * @Last Modified time: 2023-12-09 14:32:32 
#  */
import numpy as np
import torch
import torch.nn as nn
import random

seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)

class Network_tanh_per(nn.Module):

    def __init__(self, L:int, num_L:int, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_per, self).__init__()
        #
        self.device = kwargs['device']
        self.L = L
        self.M = num_L
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def input_encoding(self, t, x):
        w = 2.0 * np.pi / self.L
        k = torch.arange(1, self.M + 1).to(self.device)
        q = torch.tensor(1).to(self.device).repeat(t.shape[0],1)
        out = torch.hstack([t, q,
                         torch.cos(k * w * x).view(t.shape[0],-1), torch.sin(k * w * x).view(t.shape[0],-1)])
        return out

    def forward(self, t_x):
        #
        t = t_x[:,0].flatten()[:, None]
        x = t_x[:, 1:].unsqueeze(-1)
        H = self.input_encoding(t, x)
        x = self.fc_in(H)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh_persin(nn.Module):

    def __init__(self, L:int, num_L:int, d_in:int=1, d_out:int=1,
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_persin, self).__init__()
        #
        self.device = kwargs['device']
        self.L = L
        self.M = num_L
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList()
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def input_encoding(self, t, x):
        w = 2.0 * np.pi / self.L
        k = torch.arange(1, self.M + 1).to(self.device)
        q = torch.tensor(1).to(self.device).repeat(t.shape[0],1)
        out = torch.hstack([t, q, torch.cos(k * w * x).view(t.shape[0],-1)])
        return out

    def forward(self, t_x):
        #
        t = t_x[:,0].flatten()[:, None]
        x = t_x[:, 1:].unsqueeze(-1)
        H = self.input_encoding(t, x)
        x = self.fc_in(H)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh_percos(nn.Module):

    def __init__(self, L:int, num_L:int, d_in:int=1, d_out:int=1,
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_percos, self).__init__()
        #
        self.device = kwargs['device']
        self.L = L
        self.M = num_L
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList()
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def input_encoding(self, t, x):
        w = 2.0 * np.pi / self.L
        k = torch.arange(1, self.M + 1).to(self.device)
        q = torch.tensor(1).to(self.device).repeat(t.shape[0],1)
        out = torch.hstack([t, q, torch.sin(k * w * x).view(t.shape[0],-1)])
        return out

    def forward(self, t_x):
        #
        t = t_x[:,0].flatten()[:, None]
        x = t_x[:, 1:].unsqueeze(-1)
        H = self.input_encoding(t, x)
        x = self.fc_in(H)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh_per_2D(nn.Module):

    def __init__(self, L:int, num_L:int, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_per_2D, self).__init__()
        #
        self.device = kwargs['device']
        self.L = L
        self.M = num_L
        self.M_t = 0
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        # try:
        #     # assert kwargs['lb'].shape==(1,d_in)
        #     self.lb = kwargs['lb']
        #     self.ub = kwargs['ub']
        # except:
        #     self.lb = -torch.ones(1, d_in)
        #     self.ub = torch.ones(1, d_in)
        #
    def input_encoding(self, t, x, y):
        q = torch.tensor(1).to(self.device).repeat(t.shape[0],1)
        m = t.shape[0]
        w_x = 2.0 * np.pi / self.L
        w_y = 2.0 * np.pi / self.L
        k_x = torch.arange(1, self.M + 1).to(self.device)
        k_y = torch.arange(1, self.M + 1).to(self.device)
        k_xx, k_yy = torch.meshgrid(k_x, k_y, indexing='ij')
        k_xx = k_xx.flatten().to(self.device)
        k_yy = k_yy.flatten().to(self.device)
        k_t = torch.from_numpy(np.power(10.0, np.arange(0, self.M_t + 1))).to(self.device)
        out = torch.hstack([q, (k_t * t).view(m,-1),
                         torch.cos(k_x * w_x * x).view(m,-1), torch.cos(k_y * w_y * y).view(m,-1),
                         torch.sin(k_x * w_x * x).view(m,-1), torch.sin(k_y * w_y * y).view(m,-1),
                         torch.cos(k_xx * w_x * x) * torch.cos(k_yy * w_y * y).view(m,-1),
                         torch.cos(k_xx * w_x * x) * torch.sin(k_yy * w_y * y).view(m,-1),
                         torch.sin(k_xx * w_x * x) * torch.cos(k_yy * w_y * y).view(m,-1),
                         torch.sin(k_xx * w_x * x) * torch.sin(k_yy * w_y * y).view(m,-1)])
        return out.to(torch.float32)

    def forward(self, t_x):
        # t_x = 2. * (t_x-self.lb) / (self.ub- self.lb) - 1.
        #
        t = t_x[:,0].flatten()[:, None]
        x = t_x[:, 1].unsqueeze(-1)
        y = t_x[:, 2].unsqueeze(-1)
        H = self.input_encoding(t, x, y)
        x = self.fc_in(H)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x))+x
        x_out = x

        return self.fc_out(x_out)   

class Network_tanh_sin_per(nn.Module):

    def __init__(self, L:int, num_L:int, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_sin_per, self).__init__()
        #
        self.device = kwargs['device']
        self.L = L
        self.M = num_L
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def fun_sin(self, x):
        '''
        '''
        return torch.sin(np.pi * (x+1.))
    
    def input_encoding(self, t, x):
        w = 2.0 * np.pi / self.L
        k = torch.arange(1, self.M + 1).to(self.device)
        q = torch.tensor(1).to(self.device).repeat(t.shape[0],1)
        out = torch.hstack([t, q,
                         torch.cos(k * w * x).view(t.shape[0],-1), torch.sin(k * w * x).view(t.shape[0],-1)])
        return out

    def forward(self, t_x):
        #
        t = t_x[:,0].flatten()[:, None]
        x = t_x[:, 1:].unsqueeze(-1)
        H = self.input_encoding(t, x)
        x = self.fc_in(H)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(self.fun_sin(fc_hidden(x))) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh_sin(nn.Module):

    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_sin, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def fun_sin(self, x):
        '''
        '''
        return torch.sin(np.pi * (x+1.))

    def forward(self, x):
        #
        x = self.fc_in(x)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(self.fun_sin(fc_hidden(x))) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def forward(self, x):
        #
        # x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.activation(self.fc_in(x))
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_sin(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_sin, self).__init__()
        #
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def fun_sin(self, x):
        '''
        '''
        return torch.sin(np.pi * x)

    def forward(self, x):
        #
        # x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.fun_sin(self.fc_in(x))
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.fun_sin(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_relu(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_relu, self).__init__()
        #
        self.activation = torch.nn.ReLU()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def forward(self, x):
        #
        # x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.fc_in(x)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(torch.sin(np.pi * x)) ) + x

        return self.fc_out(x)
    
class Model():
    '''
    '''
    def __init__(self, model_type:str, device=None, dtype:torch.dtype=torch.float32):
        self.model_type = model_type
        self.device = device
        torch.set_default_dtype(dtype)
    
    def get_model(self, d_in:int=1, d_out:int=1, h_size:int=200, h_layers:int=3, **kwargs):
        if self.model_type=='sin':
            return Network_sin(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                               hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='relu':
            return Network_relu(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh':
            return Network_tanh(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_sin':
            return Network_tanh_sin(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_sin_per':
            return Network_tanh_sin_per(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_per':
            return Network_tanh_per(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_per_2D':
            return Network_tanh_per_2D(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_persin':
            return Network_tanh_persin(d_in=d_in, d_out=d_out, hidden_size=h_size,
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_percos':
            return Network_tanh_percos(d_in=d_in, d_out=d_out, hidden_size=h_size,
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        else:
            raise NotImplementedError(f'No network model {self.model_type}.')