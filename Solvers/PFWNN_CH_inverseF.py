# /*
#  * @Author: chang.ma 
#  * @Date: 2023-12-19 14:32:32
#  * @Last Modified by: chang.ma 
#  * @Last Modified time: 2023-12-19 14:32:32 
#  */
import numpy as np 
import scipy.io
import time
import os
import torch
#
from Network.FeedForward import Model
from Utils.Error import Error
# 
from Utils.GenData_Time import GenData
from Problems.Module_Time import Problem
import Solvers.Module as Module
#
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda")
print('Using {} device'.format(device))

try:
    print(f'{torch.cuda.get_device_name(0)}')
except:
    pass

class PFWNN(Module.Solver):    
    '''
    '''
    def __init__(self, Problem:Problem, 
                 Num_particles:int, Num_tin_size:int, Nx_integral:int, 
                 train_xbd_size_each_face:int, train_tbd_size:int,
                 train_init_size:int, R_max:float, maxIter:int, 
                 lr:float, net_type_u:str,net_type_E:str, net_type_mu:str, **kwargs):
        #
        self.Num_particles = Num_particles
        self.Num_tin_size = Num_tin_size
        self.Nx_integral = Nx_integral
        self.train_xbd_size_each_face = train_xbd_size_each_face
        self.train_tbd_size = train_tbd_size
        self.train_init_size = train_init_size
        self.Rmax = R_max
        self.iters = maxIter
        self.lr = lr
        self.net_type_u = net_type_u 
        self.net_type_F= net_type_E 
        self.net_type_mu = net_type_mu
        # Other settings
        self.Nt_integral = kwargs['Nt_integral']
        self.Rway = kwargs['R_way']
        self.Rmin = kwargs['R_min']
        self.w_init = kwargs['w_init']
        self.w_weak = kwargs['w_weak']
        # data 
        self.noise_level = kwargs['noise_level'] 
        self.w_data = kwargs['w_data']
        self.topk = kwargs['topk']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.hidden_n_E = kwargs['hidden_width_E']
        self.hidden_l_E = kwargs['hidden_layer_E']
        self.dtype = kwargs['dtype']
        self.lrDecay = 1.
        # timestep       
        self.tol = kwargs['tol']
        self.M =  torch.tensor(np.triu(np.ones((Num_tin_size, Num_tin_size)), k=1).T).to(torch.float32).to(device)
        #
        self.problem = Problem(M=kwargs['coe_M'], lambda2_M=kwargs['coe_lam2M'], loadpath=kwargs['data_path'],test_type=kwargs['test_fun'], dtype=self.dtype['numpy'])
        self.data = GenData(self.problem, dtype=self.dtype['numpy'])
        # network
        self.num_L = 5
        if self.problem.dim == 1:
            self.L = 2.
        else:
            self.L = 1.
        #
        self.t1 = kwargs['t1']
        self.t2 = kwargs['t2']
        self.interval = kwargs['interval']

    def _save(self, save_path:str, model_type:str)->None:
        '''
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 
        if model_type=='model_final':
            dict_loss = {}
            dict_loss['loss'] = self.loss_u_list
            dict_loss['error_u'] = self.error_u
            dict_loss['error_E'] = self.error_E
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+'loss_error_saved.mat', dict_loss)
            
        if model_type=='model_best_loss':
            dict_loss = {}
            dict_loss['loss'] = self.loss_u_list
            dict_loss['error_u'] = self.error_u
            dict_loss['error_E'] = self.error_E
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+'loss_error_saved_best.mat', dict_loss)
        # 
        model_dict = {'model_u':self.model_u.state_dict(), 'model_E':self.model_E.state_dict()}
        torch.save(model_dict, save_path+f'trained_{model_type}.pth')


    def _load(self, load_path:str, model_type:str='model_best_loss')->None:
        '''
        '''
        model_dict = torch.load(load_path+f'trained_{model_type}.pth')
        try:
            self.model_u.load_state_dict(model_dict['model_u'])
            self.model_E.load_state_dict(model_dict['model_E'])
        except:
            self.get_net()
            self.model_u.load_state_dict(model_dict['model_u'])
            self.model_E.load_state_dict(model_dict['model_E'])

    def test(self, save_path:str, model_type='model_best_loss')->None:
        '''
        '''
        # load the trained model
        self._load(save_path, model_type)
        #
        u_test, x_test, t_test, _, _, _, _, _, _ = \
            self.problem._valid_u_timestep(t1=self.t1, t2=self.t2, interval=self.interval)
        E_test = self.problem._fun_para(u_test)
        with torch.no_grad():
            u_pred = self.model_u(torch.cat([t_test.to(device), x_test.to(device)], dim=1))
            E_pred = self.model_E(u_pred.detach().requires_grad_(True).to(device))
            E_pred_1 = self.model_E(u_test.to(device))
        # 
        dict_test = {}
        dict_test['x_test'] = x_test.detach().cpu().numpy()
        dict_test['t_test'] = t_test.detach().cpu().numpy()
        dict_test['u_test'] = u_test.detach().cpu().numpy()
        dict_test['u_pred'] = u_pred.detach().cpu().numpy()
        dict_test['E_test'] = E_test.detach().cpu().numpy()
        dict_test['E_pred'] = E_pred.detach().cpu().numpy()
        dict_test['E_pred_1'] = E_pred.detach().cpu().numpy()
        #
        scipy.io.savemat(save_path+'test_saved.mat', dict_test)

    def get_net(self)->None:
        '''
        实例化
        '''
        if self.problem.dim ==1:
            d_in = self.num_L * self.problem.dim * 2 + 2
        else:
            d_in = 2 + self.num_L * 4 + self.num_L**2 * 4
        #
        kwargs = {'d_in':  d_in,
                  'L': self.L,
                  'num_L': self.num_L,
                  'h_size': self.hidden_n,
                  'h_layers': self.hidden_l,
                  'device':device
                  }
        self.model_u = Model(self.net_type_u, device, dtype=self.dtype['torch']).get_model(**kwargs) 
        #
        kwargs_mu = {'d_in': d_in,
                  'L': self.L,
                  'num_L': self.num_L,
                  'h_size': self.hidden_n,
                  'h_layers': self.hidden_l,
                  'device':device
                  }
        self.model_mu = Model(self.net_type_mu, device, dtype=self.dtype['torch']).get_model(**kwargs_mu) 
        #
        kwargs_E = {'d_in': 1 ,
                  'h_size': self.hidden_n_E,
                  'h_layers': self.hidden_l_E}
        self.model_E = Model(self.net_type_F, device, dtype=self.dtype['torch']).get_model(**kwargs_E)
        # 
        self.optimizer = torch.optim.Adam([
            {'params': self.model_u.parameters(), 'lr': self.lr}, 
            {'params': self.model_mu.parameters(), 'lr': self.lr},
            {'params': self.model_E.parameters(), 'lr': self.lr}
            ])
        #
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=(1.-self.lrDecay/self.iters), last_epoch=-1)

    def get_loss(self, **args):
        '''
        '''
        ########### Residual inside the domain
        x_scaled = self.data.get_x_scaled(Nx_scaled=self.Nx_integral, method=self.int_method)
        #
        R, xc, tc = self.data.get_txc(t1=self.t1, t2=self.t2, N_xc=self.Num_particles,
                                      Nt_size=self.Num_tin_size, R_max=self.Rmax, R_min=self.Rmin)
        #
        weak1, weak2 = self.problem.weak_mu_invF(self.model_u, self.model_mu, self.model_E, x_scaled.to(device), xc.to(device), tc.to(device), R.to(device), device=device) 
        weak1_form = (weak1 ** 2).view(self.Num_tin_size, self.Num_particles)
        L1_t= torch.mean(weak1_form, dim=1)
        W1 = torch.exp(-self.tol * (self.M @ L1_t)).data.to(device)
        loss_equ = torch.mean( W1 * L1_t )* self.w_weak
        #
        weak2_form = (weak2 ** 2).view(self.Num_tin_size, self.Num_particles)
        L2_t= torch.mean(weak2_form, dim=1)
        W2 = torch.exp(-self.tol * (self.M @ L2_t)).data.to(device)
        loss_equ += torch.mean( W2 * L2_t )* self.w_weak
        ########### mismatch at inital time
        u_init_true, x_init, t_init = self.problem.fun_u_init()
        u_init_pred = self.model_u(torch.cat([t_init.to(device), x_init.to(device)], dim=1))
        loss_init = torch.mean( (u_init_pred - u_init_true.to(device)) **2 ) * self.w_init
        ########## mismatch inside the domain (data loss in inverse problem)
        # u_valid, x_valid, t_valid = self.problem._fun_u_portion(self.portion_t, self.portion_x)
        u_valid, x_valid, t_valid, _, _, _, _, _, _ = \
            self.problem._valid_u_timestep(t1=self.t1, t2=self.t2, interval=self.interval)
        u_pred_in = self.model_u(torch.cat([t_valid.to(device), x_valid.to(device)], dim=1))
        u_mea_in = u_valid.to(device) + torch.randn_like(u_pred_in) * self.noise_level
        loss_data = torch.mean((u_pred_in - u_mea_in)**2 )*self.w_data
        loss = loss_equ + loss_init + loss_data

        return loss, loss_equ, loss_init, loss_data, W1

    def train(self, save_path:str)->None:
        '''
        Train the network
        '''
        t_start = time.time()
        self.get_net()
        # 
        u_valid, x_valid, t_valid, _, _, _, _, _, _ = \
            self.problem._valid_u_timestep(t1=self.t1, t2=self.t2, interval=self.interval)
        E_valid= self.problem._fun_para(u_valid)
        # 
        iter = 0
        best_loss = 1e10
        self.time_list = []
        self.loss_u_list = []
        self.loss_equ_list = []
        self.loss_init_list = []
        self.loss_data_list = []
        self.error_u, self.error_M, self.error_E = [], [], []
        for iter in range(self.iters):
            if self.Rway=='Rfix':
                R_adaptive = self.Rmax 
            elif self.Rway=='Rascend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * iter/self.iters
            elif self.Rway=='Rdescend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * (1-iter/self.iters)
            loss_u_train, loss_equ_train, loss_init_train, loss_data_train, W = self.get_loss(**{'Rmax':R_adaptive, 'Rmin':R_adaptive})
            # Train the network
            self.optimizer.zero_grad()
            loss_u_train.backward()
            self.optimizer.step()
            self.scheduler.step()
            # Save loss and error
            iter += 1
            self.loss_u_list.append(loss_u_train.item())
            self.loss_equ_list.append(loss_equ_train.item())
            self.loss_init_list.append(loss_init_train.item())
            self.loss_data_list.append(loss_data_train.item())
            self.time_list.append(time.time()-t_start)
            with torch.no_grad():
                u_pred_valid = self.model_u(torch.cat([t_valid.to(device), x_valid.to(device)], dim=1))
                E_pred_valid = self.model_E(u_pred_valid.detach().requires_grad_(True).to(device))
                error_u_valid = Error().L2_error(u_pred_valid, u_valid.to(device))
                error_E_valid = Error().L2_error(E_pred_valid, E_valid.to(device))
                self.error_u.append(error_u_valid)
                self.error_E.append(error_E_valid)
                # Save network model (best loss)
                if (error_E_valid) < best_loss:
                    best_loss = error_E_valid
                    self._save(save_path, model_type='model_best_loss')
                if iter%100 == 0:
                    print(f"Iter: {iter+1},error_u:{self.error_u[-1]:.4f},error_E:{self.error_E[-1]:.4f}, loss_u:{np.mean(self.loss_u_list[-50:]):.4f},loss_equ:{np.mean(self.loss_equ_list[-50:]):.8f}, loss_init:{np.mean(self.loss_init_list[-50:]):.8f}, loss_data:{np.mean(self.loss_data_list[-50:]):.8f},weigh-1:{W[-1]},R:{R_adaptive:.4f},times:{self.time_list[-1]}")
        # Save network model (final)
        self._save(save_path, model_type='model_final')
        print(f'The total time is {time.time()-t_start:.4f}')