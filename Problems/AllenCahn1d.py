# /*
#  * @Author: chang.ma 
#  * @Date: 2023-12-09 14:32:32
#  * @Last Modified by: chang.ma 
#  * @Last Modified time: 2023-12-09 14:32:32 
#  */
import numpy as np 
import torch 
import sys
import os
import scipy.io
from torch.autograd import Variable
#
from Utils.TestFun import TestFun
import Problems.Module_Time as Module
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class Problem(Module.Problem):
    '''
    The Allen-Cahn equation:
          u_t - lambda * u_xx + 5u^3 - 5u = f(t,x)  in [0,T]*[-1,1]
          u(0,x)   = g(x)   in [-1,1]
          u(t,-1) = u(t,1)
          u_x(t,-1) = u_x(t,1)
    where
          g(x) = x^2 * cos(pi * x)
    '''
    def __init__(self, loadpath, M, lambda2_M, test_type:str='Wendland', dtype:np.dtype=np.float32):
        #
        self._dim = 1
        self._name = 'AC_1d'
        self._t0 = 0.
        self._tT = 1.
        #
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        self._k = torch.pi
        self._lambda2_M = lambda2_M
        self.M = M
        #
        self._dtype = dtype
        self._test_fun = TestFun(f'{test_type}', self.dim)
        #
        self._true_data = scipy.io.loadmat(loadpath)

    @property
    def name(self):
        return self._name
    
    @property
    def dim(self):
        return self._dim

    @property
    def lb(self):
        return np.array([self._t0, self._lb[0]]).reshape(-1, 1+self.dim).astype(self._dtype)

    @property
    def ub(self):
        return np.array([self._tT, self._ub[0]]).reshape(-1, 1+self.dim).astype(self._dtype)
    
    def _fun_u(self, x:torch.tensor=None, t:torch.tensor=None):
        '''
        The ground truth of u 
        Input: x:size(?,d)
               t:size(?,1)
        Return: u: size(?,1)
                    or
                u: size(?,1)
                x: size(?,d)
                t: size(?,1)
        '''
        if x is not None:
            raise NotImplementedError('Explicit solution is not available.')
        else:
            t_mesh = self._true_data['tt'].flatten()
            x_mesh = self._true_data['x'].flatten()
            t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
            #
            t = torch.from_numpy(t_mesh.reshape(-1,1).astype(self._dtype))
            x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
            #
            u = torch.from_numpy(self._true_data['uu'].reshape(-1,1).astype(self._dtype))
            
            return u, x, t
    
    def _fun_u_portion(self,portion_t,portion_x):
        
                   
        t_mesh = self._true_data['tt'].flatten()[0:201:portion_t]
        x_mesh = self._true_data['x'].flatten()[0:513:portion_x]
        
        t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
        #
        t = torch.from_numpy(t_mesh.reshape(-1,1).astype(self._dtype))
        x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
        #
        u = self._true_data['uu'][0:513:portion_x, 0:201:portion_t]
        u = torch.from_numpy(u.reshape(-1,1).astype(self._dtype))
        
        return u, x, t
       
    def _valid_u_timestep(self, t1, t2, interval, x:torch.tensor=None, t:torch.tensor=None):
        '''
        The ground truth of u
        Input: x:size(?,d)
               t:size(?,1)
        Return: u: size(?,1)
                    or
                u: size(?,1)
                x: size(?,d)
                t: size(?,1)
        '''
        if x is not None:
            raise NotImplementedError('Explicit solution is not available.')
        else:
            t_mesh = self._true_data['tt'].flatten()[int(t1*interval):int(t2*interval)]
            x_mesh = self._true_data['x'].flatten()
            t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
            #
            t = torch.from_numpy(t_mesh.reshape(-1,1).astype(self._dtype))
            x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
            #
            u = torch.from_numpy(self._true_data['uu'][:,int(t1*interval):int(t2*interval)].reshape(-1,1).astype(self._dtype))

            # 下一个阶段的initial
            t0_mesh = self._true_data['tt'].flatten()[int(t2*interval)]
            x0_mesh = self._true_data['x'].flatten()
            t0_mesh, x0_mesh = np.meshgrid(t0_mesh, x0_mesh)
            #
            t0 = torch.from_numpy(t0_mesh.reshape(-1,1).astype(self._dtype))
            x0 = torch.from_numpy(x0_mesh.reshape(-1,1).astype(self._dtype))
            #
            u0 = torch.from_numpy(self._true_data['uu'][:,int(t2*interval)].reshape(-1,1).astype(self._dtype))

            # 前面阶段的initial
            tpre_mesh = self._true_data['tt'].flatten()[:int(t2*interval)]
            xpre_mesh = self._true_data['x'].flatten()
            tpre_mesh, xpre_mesh = np.meshgrid(tpre_mesh, xpre_mesh)
            #
            tpre = torch.from_numpy(tpre_mesh.reshape(-1,1).astype(self._dtype))
            xpre = torch.from_numpy(xpre_mesh.reshape(-1,1).astype(self._dtype))
            #
            upre = torch.from_numpy(self._true_data['uu'][:,:int(t2*interval)].reshape(-1,1).astype(self._dtype))

            return u, x, t, u0, x0, t0, upre, xpre, tpre

        ## MBP
        # t_mesh = self._true_data['tt'].flatten()[0:201:2]
        # x_mesh = self._true_data['x'].flatten()[0:513:2]
        #
        # t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
        # #
        # t = torch.from_numpy(t_mesh.reshape(-1, 1).astype(self._dtype))
        # x = torch.from_numpy(x_mesh.reshape(-1, 1).astype(self._dtype))
        # #
        # u = self._true_data['uu'][0:513:2, 0:201:2]
        # u = torch.from_numpy(u.reshape(-1, 1).astype(self._dtype))
        #
        # return u, x, t, u, x, t, u, x, t

    def fun_u_init(self)->torch.tensor:
        '''
        Input: x: size(?,d)
               t: size(?,1)
        Output: u: size(?,1)
        '''
        # assert len(x.shape)==2 and len(t.shape)==2
        # return x**2 * torch.cos(self._k * x)
        t_mesh = self._true_data['tt'].flatten()[0]
        x_mesh = self._true_data['x'].flatten()
        t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
        #
        t = torch.from_numpy(t_mesh.reshape(-1,1).astype(self._dtype))
        x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
        #
        u = torch.from_numpy(self._true_data['uu'][:,0].reshape(-1,1).astype(self._dtype))
        
        return u, x, t
    
    def fun_u_bd(self, model_u, x_lb, x_ub, t:torch.tensor )->torch.tensor:
        '''
        Input:   x_list: list= [size(n,1)]*2d
                 t: size(n,1)
                 model       
        Output:  u_lb, u_ub: size(n*2d,1) if model_u is None
                 u_lb_nn, u_ub_nn: size(n*2d,1) if model_u is given.
        '''
    
            #
        x_lb = Variable(x_lb.view(-1,1), requires_grad=True)
        x_ub = Variable(x_ub.view(-1,1), requires_grad=True)
        u_lb_nn = model_u(torch.cat([t, x_lb], dim=1))
        u_ub_nn = model_u(torch.cat([t, x_ub], dim=1))
        #
        du_lb_nn = self._grad_u(x_lb, u_lb_nn) 
        du_ub_nn = self._grad_u(x_ub, u_ub_nn)

        return u_lb_nn - u_ub_nn, du_lb_nn - du_ub_nn

    def _fun_para(self, u):
        '''
        '''
        return u**3 - u

    def fun_f(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        Input: x: size(?,d)
               t: size(?,1)
        Output: f: size(?,1)
        '''
        assert len(x.shape)==2 and len(t.shape)==2
        return torch.zeros_like(t)
    
    def strong(self, model_u, x:torch.tensor, t:torch.tensor)->torch.tensor:

        ############ variables
        x = Variable(x , requires_grad=True)
        t = Variable(t, requires_grad=True)
        ############# grads 
        u = model_u(torch.cat([t,x], dim=1))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        du_d2x = self._Laplace_u([x], du_dx)
        ############# The pde
        left = du_dt+ self.M * (u**3 - u) - self._lambda2_M * du_d2x
        ############## The right hand side
        right = self.fun_f(x, t)
        
        return left-right
      
    def weak(self, model_u, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
             R:torch.tensor)->torch.tensor:
        '''
        The weak form
        Input:  model: the network model
                x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   t: size(?, 1, 1)
                   R: size(?, 1, 1)
        Output: weak_form
        '''
        ###############
        m = x_scaled.shape[0]
        x = x_scaled * R + xc  
        t = t.repeat(1,m,1)
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        u = model_u(torch.cat([t,x], dim=1))   
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        u, du_dt, du_dx = u.view(-1, m, 1), du_dt.view(-1, m, 1), du_dx.view(-1, m, self.dim)
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form
        left = torch.mean(du_dt * v, dim=1)  
        left += self.M * torch.mean( (u**3 -u) * v, dim=1) 
        left += self._lambda2_M * torch.mean(torch.sum(du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        f = self.fun_f(x=x, t=t).view(-1, m, 1)
        right = torch.mean( f * v, dim=1)

        return left-right
    
    def weak_inverseF(self, model_u, model_E, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
             R:torch.tensor, device)->torch.tensor:
        '''
        The weak form
        Input:  model: the network model
                x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   t: size(?, 1, 1)
                   R: size(?, 1, 1)
        Output: weak_form
        '''
        ###############
        m = x_scaled.shape[0]
        x = x_scaled * R + xc  
        t = t.repeat(1,m,1)
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        u = model_u(torch.cat([t,x], dim=1))   
        E = model_E(u.detach().requires_grad_(True).to(device))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        u, du_dt, du_dx = u.view(-1, m, 1), du_dt.view(-1, m, 1), du_dx.view(-1, m, self.dim)
        E = E.view(-1, m, 1)
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form 
        left = torch.mean(du_dt * v, dim=1)  
        left += self.M * torch.mean( E * v, dim=1) 
        left += self._lambda2_M * torch.mean(torch.sum( du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        f = self.fun_f(x=x, t=t).view(-1, m, 1)
        right = torch.mean( f * v, dim=1)

        return left-right