#  * @Author: chang.ma
import numpy as np 
import scipy.io
import time
import os
import torch
#
from Network.FeedForward import Model
from Utils.Error import Error
from Utils.GenData_Time import GenData
from Problems.Module_Time import Problem
import Solvers.Module as Module
#
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
device = torch.device("cuda")
print('Using {} device'.format(device))
try:
    print(f'{torch.cuda.get_device_name()}')
except:
    pass


class PFWNN(Module.Solver):
    '''
    '''
    def __init__(self, Problem:Problem, 
                 Num_particles:int, Num_tin_size:int, Nx_integral:int, 
                 train_xbd_size_each_face:int, train_tbd_size:int,
                 train_init_size:int, R_max:float, maxIter:int, 
                 lr:float, net_type:str, **kwargs):
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
        self.net_type = net_type
        # Other settings
        self.Nt_integral = kwargs['Nt_integral']
        self.Rway = kwargs['R_way']
        self.Rmin = kwargs['R_min']
        self.w_init = kwargs['w_init']
        self.w_weak = kwargs['w_weak']
        self.w_bd = kwargs['w_bd']
        self.topk = kwargs['topk']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.dtype = kwargs['dtype']
        self.lrDecay = 1.
        #
        self.problem = Problem(loadpath=kwargs['loadpath'], M =kwargs["coe_M"], lambda2_M =kwargs["coe_lam2M"],
                               test_type=kwargs['test_fun'], dtype=self.dtype['numpy'])
        self.data = GenData(self.problem, dtype=self.dtype['numpy'])
        #s
        self.tol = kwargs['tol']
        self.M =  torch.tensor(np.triu(np.ones((Num_tin_size, Num_tin_size)), k=1).T).to(torch.float32).to(device)
        self.num_L = 5
        if self.problem.dim == 1:
            self.L = 2.
        else:
            self.L = 1.
        #
        self.t1 = kwargs['t1']
        self.t2 = kwargs['t2']
        self.stage = kwargs['stage']
        self.step = kwargs['step']
        self.interval = kwargs['interval']
        self.saved_path = kwargs['saved_path']
        #
        self.Nc_append = kwargs['Nc_append']
        self.ada_step = kwargs['ada_step']
        
    def _save(self, model_type:str)->None:
        '''
        '''
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)
        # 
        if model_type=='model_final':
            dict_loss = {}
            dict_loss['loss_u'] = self.loss_u_list
            dict_loss['error'] = self.error
            dict_loss['time'] = self.time_list
            scipy.io.savemat(self.saved_path + 'loss_error_saved.mat', dict_loss)
        # 
        torch.save(self.model_u.state_dict(), self.saved_path+f'trained_{model_type}.pth')

    def _load(self, load_path:str, model_type:str='model_best_loss')->None:
        '''
        '''
        try:
            self.model_u.load_state_dict(torch.load(load_path+f'trained_{model_type}.pth'))
        except:
            self.get_net()
            self.model_u.load_state_dict(torch.load(load_path+f'trained_{model_type}.pth'))

    def test(self, step:int, stage:int, model_type='model_best_loss')->None:
        '''
        '''
        stage = self.stage
        step = self.step
        # load the trained model
        self._load(self.saved_path, model_type)
        #
        u_test, x_test, t_test, u0_test, x0_test, t0_test, upre_valid, xpre_valid, tpre_valid  = \
            self.problem._valid_u_timestep(self.t1, self.t2, self.interval)
        with torch.no_grad():
            u_pred = self.model_u(torch.cat([t_test.to(device), x_test.to(device)], dim=1))
            state0 = self.model_u(torch.cat([t0_test.to(device), x0_test.to(device)], dim=1))

        dict_test = {}
        #
        if self.problem.dim == 1:
            shape_dim = 512
        else:
            shape_dim = 16384
       
        dict_test['x_test_preall'] = xpre_valid.detach().cpu().numpy()
        dict_test['t_test_preall'] = tpre_valid.detach().cpu().numpy()
        dict_test['u_test_preall'] = upre_valid.detach().cpu().numpy()
        if self.stage == 0:
            dict_test['u_pred_preall'] = u_pred.detach().cpu().numpy()
        else:
            # 返回上一级目录
            parent_directory_path = os.path.dirname(os.path.dirname(self.saved_path))
            u_pred_preall = scipy.io.loadmat(parent_directory_path +'/'+ str(self.stage-1)+f'/test_saved.mat')['u_pred_preall']
            u_pred_preall = torch.from_numpy(u_pred_preall.reshape(shape_dim,step*stage)).to(device)
            u_pred_preall = torch.cat((u_pred_preall, u_pred.reshape(shape_dim,step)), dim =1)
            dict_test['u_pred_preall'] = u_pred_preall.reshape(-1,1).detach().cpu().numpy()
        # 保存最后一个时刻
        dict_test['x0_test'] = x0_test.detach().cpu().numpy()
        dict_test['t0_test'] = t0_test.detach().cpu().numpy()
        dict_test['state0'] = state0.detach().cpu().numpy()
        scipy.io.savemat(self.saved_path+f'test_saved.mat', dict_test)
        print("I have saved")

    def get_net(self)->None:
        '''
        '''
        if self.problem.dim ==1:
            d_in = self.num_L * self.problem.dim * 2 + 2
        else:
            d_in = 2 + self.num_L * 4 + self.num_L**2 * 4
            
        kwargs = {'d_in': d_in,
                  'L': self.L,
                  'num_L': self.num_L,
                  'h_size': self.hidden_n,
                  'h_layers': self.hidden_l,
                  'device':device}
        self.model_u = Model(self.net_type, device, dtype=self.dtype['torch']).get_model(**kwargs)
        # 
        self.optimizer_u = torch.optim.Adam(self.model_u.parameters(), lr=self.lr)
        #
        self.scheduler_u = torch.optim.lr_scheduler.StepLR(
            self.optimizer_u, 1, gamma=(1.-self.lrDecay/self.iters), last_epoch=-1)

    def get_loss(self,x_scaled, xc, tc, R, w_weak, w_init, **args):
        '''
        '''
        ########### Residual inside the domain
        # CH
        weak1, weak2 = self.problem.weak(self.model_u, x_scaled.to(device), xc.to(device), tc.to(device), R.to(device))
        weak1_form_0 = (weak1 ** 2).view(self.Num_tin_size, self.Num_particles)
        # self-adaptive weight
        sums = weak1_form_0.sum(dim=1, keepdim=True).detach()
        weight = weak1_form_0.detach() / sums * self.Num_particles
        weak1_form = weak1_form_0 * weight
        #
        L1_t= torch.mean(weak1_form, dim=1)
        W1 = torch.exp(-self.tol * (self.M @ L1_t)).data.to(device)
        loss_equ = torch.mean( W1 * L1_t )* w_weak
        ##################################
        ##################################
        weak2_form_0 = (weak2 ** 2).view(self.Num_tin_size, self.Num_particles)
        # self-adaptive weight
        sums = weak2_form_0.sum(dim=1, keepdim=True).detach()
        weight = weak2_form_0.detach() / sums * self.Num_particles
        weak2_form = weak2_form_0 * weight
        #
        L2_t= torch.mean(weak2_form, dim=1)
        W2 = torch.exp(-self.tol * (self.M @ L2_t)).data.to(device)
        loss_equ += torch.mean( W2 * L2_t )* w_weak
        ########### mismatch at inital time
        if self.stage == 0:
            u_init_true, x_init, t_init= self.problem.fun_u_init( )
            u_init_pred = self.model_u(torch.cat([t_init.to(device), x_init.to(device)], dim=1))
            u_init_true = u_init_true.to(device)
        else:
            parent_directory_path = os.path.dirname(os.path.dirname(self.saved_path))
            pre_stage = parent_directory_path + '/' + str(self.stage-1)+ f'/test_saved.mat'
            u_init_true = scipy.io.loadmat(pre_stage)['state0']
            t_init, x_init =  scipy.io.loadmat(pre_stage)['t0_test'], scipy.io.loadmat(pre_stage)['x0_test']
            t_init, x_init = torch.from_numpy(t_init).to(device), torch.from_numpy(x_init).to(device)
            u_init_true = torch.from_numpy(u_init_true).to(device)
            u_init_pred = self.model_u(torch.cat([t_init.to(device), x_init.to(device)], dim=1))
        loss_init = torch.mean( (u_init_pred - u_init_true) **2 ) * w_init #self.w_init
        loss = loss_equ + loss_init

        return loss, loss_equ, loss_init, W1, weak2_form_0

    def train(self)->None:
        '''
        Train the network
        '''
        t_start = time.time()
        self.get_net()
        # 
        u_valid, x_valid, t_valid, u0_valid, x0_valid, t0_valid, upre_valid, xpre_valid, tpre_valid = \
            self.problem._valid_u_timestep(t1=self.t1, t2=self.t2, interval=self.interval)
        
        # 
        iter = 0
        best_loss = 1e50
        self.time_list = []
        self.loss_u_list = []
        self.loss_equ_list = []
        self.loss_init_list = []
        self.error = []
        self.error_preall = []

        #data generate
        x_scaled = self.data.get_x_scaled(Nx_scaled=self.Nx_integral, method=self.int_method)
        #
        R, xc, tc = self.data.get_txc(t1=self.t1, t2=self.t2, N_xc=self.Num_particles, Nt_size=self.Num_tin_size,
                                      R_max=self.Rmax, R_min=self.Rmin)

        for iter in range(self.iters):
            if self.Rway=='Rfix':
                R_adaptive = self.Rmax 
            elif self.Rway=='Rascend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * iter/self.iters
            elif self.Rway=='Rdescend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * (1-iter/self.iters)
            #
            if iter <= 5000:
                w_weak = 1.
                w_init = 10000.
            elif 5000 < iter <= 10000:
                w_weak = 10.
                w_init = 10000.
            else:
                w_weak = 100.
                w_init = 10000.
                
            loss_u_train, loss_equ_train, loss_init_train, W, weak_residual = (
                self.get_loss(x_scaled, xc, tc, R, w_weak, w_init, **{'Rmax':R_adaptive, 'Rmin':R_adaptive}))
            # Train the network
            self.optimizer_u.zero_grad()
            loss_u_train.backward()
            self.optimizer_u.step()
            self.scheduler_u.step()
            # Save loss and error
            iter += 1
            self.loss_u_list.append(loss_u_train.item())
            self.loss_equ_list.append(loss_equ_train.item())
            self.loss_init_list.append(loss_init_train.item())
            self.time_list.append(time.time()-t_start)
            with torch.no_grad():
                u_pred_valid = self.model_u(torch.cat([t_valid.to(device), x_valid.to(device)], dim=1))
                error_valid = Error().L2_error(u_pred_valid, u_valid.to(device))
                self.error.append(error_valid)
                # self-adaptive
                if iter % self.ada_step == 0:
                    if self.problem.dim == 1:
                        _, top_indices = torch.topk(weak_residual, k=self.Nc_append, dim=1)
                        top_indices = top_indices.detach().cpu()
                        xc = xc.view(self.Num_tin_size, self.Num_particles).detach().cpu()
                        tc = tc.view(self.Num_tin_size, self.Num_particles).detach().cpu()
                        top_xc_0 = torch.gather(xc, dim=1, index=top_indices)
                        top_xc = top_xc_0.view(-1, 1, 1)
                        top_tc_0 = torch.gather(tc, dim=1, index=top_indices)
                        top_tc = top_tc_0.view(-1, 1, 1)
                        #
                        _, xc, tc = self.data.get_txc(t1=self.t1, t2=self.t2,
                                                      N_xc=(self.Num_particles - self.Nc_append),
                                                      Nt_size=self.Num_tin_size, R_max=self.Rmax, R_min=self.Rmin)
                        xc = torch.cat((xc, top_xc), dim=0)
                        tc = torch.cat((tc, top_tc), dim=0)
                        #
                        dict_ada = {}
                        dict_ada['ada_x'] = top_xc_0.numpy()
                        dict_ada['ada_t'] = top_tc_0.numpy()
                        if not os.path.exists(self.saved_path):
                            os.makedirs(self.saved_path)
                        scipy.io.savemat(self.saved_path + f'ada_saved.mat', dict_ada)
                    else:
                        _, top_indices = torch.topk(weak_residual, k=self.Nc_append, dim=1)
                        top_indices = top_indices.detach().cpu()
                        xc_app = xc.view(self.Num_tin_size, self.Num_particles, 2).detach().cpu()
                        tc_app = tc.view(self.Num_tin_size, self.Num_particles).detach().cpu()
                        top_xc_0 = torch.gather(xc_app, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 2))
                        top_xc = top_xc_0.view(-1, 1, 2)
                        top_tc_0 = torch.gather(tc_app, dim=1, index=top_indices)
                        top_tc = top_tc_0.view(-1, 1, 1)
                        #
                        _, xc, tc = self.data.get_txc(t1=self.t1, t2=self.t2,
                                                      N_xc=(self.Num_particles - self.Nc_append),
                                                      Nt_size=self.Num_tin_size, R_max=self.Rmax, R_min=self.Rmin)
                        xc = torch.cat((xc, top_xc), dim=0)
                        tc = torch.cat((tc, top_tc), dim=0)
                        #
                        dict_ada = {}
                        dict_ada['ada_x'] = top_xc_0.numpy()
                        dict_ada['ada_t'] = top_tc_0.numpy()
                        if not os.path.exists(self.saved_path):
                            os.makedirs(self.saved_path)
                        scipy.io.savemat(self.saved_path + f'ada_saved.mat', dict_ada)
                #
                if (loss_u_train.item()) < best_loss:
                    best_loss = loss_u_train.item()
                    self._save(model_type='model_best_loss')
                # 
                if iter%100 == 0:
                    print(f"Iter:{iter+1},error:{self.error[-1]:.4f},loss_u:{np.mean(self.loss_u_list[-5:]):.8f},loss_equ:{np.mean(self.loss_equ_list[-5:]):.8f},loss_init:{np.mean(self.loss_init_list[-5:]):.8f},R:{R_adaptive:.4f},weigh-1:{W[-1]},time:{self.time_list[-1]:.4f}")
                           
       
        # Save network model (final)
        self._save(model_type='model_final')
        print(f'The total time is {time.time()-t_start:.4f}')

        
        
        