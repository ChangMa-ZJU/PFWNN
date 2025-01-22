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
print(f'The Base Dir is: {BASE_DIR}')

class Problem(Module.Problem):
    '''
    The Cahn-Hillard equation:
          u_t + (0.001 * u_xxxx - 1 * (u^3 - u)_xx) = 0  in [0,T]*[-1,1]
          u(0,x)   = g(x)   in [-1,1]
          u(t,-1) = u(t,1)
          u_x(t,-1) = u_x(t,1)
    where
          (cos(pi*xx)- exp(-(pi*xx).^2)+1)./2, gamma2=0.01, gamma1=1e-6.
    '''

    def __init__(self, loadpath, M, lambda2_M, test_type:str='Wendland', dtype:np.dtype=np.float32):
        #
        self._dim = 1
        self._name = 'CH1d'
        #
        self._t0 = 0.
        self._tT = 1.
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        self._k = torch.pi
        self._lambda2 = lambda2_M
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
        #
        # d2u_lb_nn = self._Laplace_u([x_lb], du_lb_nn) 
        # d2u_ub_nn = self._Laplace_u([x_ub], du_ub_nn)

        return u_lb_nn - u_ub_nn, du_lb_nn - du_ub_nn #, d2u_lb_nn - d2u_ub_nn
    
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
        assert len(x.shape)==2 and len(t.shape)==2 # assert 断言 判断正确的话 程序继续往下进行
        return torch.zeros_like(t)
    
    def strong(self, model_u, x:torch.tensor, t:torch.tensor)->torch.tensor:
        ############ variables
        x = Variable(x, requires_grad=True)
        t = Variable(t, requires_grad=True)
        ############# grads 
        u = model_u(torch.cat([t,x], dim=1))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        du_d2x = self._Laplace_u([x], du_dx)
        mu = (u**3-u) - self._lambda2 * du_d2x
        dmu_dx = self._grad_u(x, mu)
        dmu_d2x = self._Laplace_u([x], dmu_dx)
        ############# The pde
        left = du_dt - self.M * dmu_d2x
        ############## The right hand side
        right = self.fun_f(x, t)
        
        return left-right

    # Biharmonic form 4-order 不需要
    def weak_bihar(self, model_u, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
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
        mu = u**3 - u
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        print(type(x))
        print(type([x]))
        du_d2x = self._Laplace_u([x], du_dx)
        u, du_dt, du_d2x = u.view(-1, m, 1), du_dt.view(-1, m, 1), du_d2x.view(-1, m, 1)   
        dmu = self._grad_u(x, mu)
        dmu = dmu.view(-1, m, self.dim)
        #
        v, dv_scaled, Lv_scaled = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        Lv = Lv_scaled/(R**2)
        ################## weak form
        left = torch.mean(du_dt * v, dim=1)  
        left += self.M * self._lambda * torch.mean( du_d2x * Lv, dim=1) 
        left += self.M * torch.mean(torch.sum( dmu * dv, dim=2, keepdims=True), dim=1)
        # 
        f = self.fun_f(x=x, t=t).view(-1, m, 1)
        right = torch.mean(f * v, dim=1)

        return left-right
    
    # Intermediate function mu
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
        du_d2x = self._Laplace_u([x], du_dx)
        mu = (u**3-u) - self._lambda2 * du_d2x
        dmu_dx = self._grad_u(x, mu)
        u, mu, du_dt, dmu_dx = u.view(-1, m, 1), mu.view(-1, m, 1), du_dt.view(-1, m, 1), dmu_dx.view(-1, m, self.dim)
        du_dx = du_dx.view(-1, m, self.dim)      
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form
        left1 = torch.mean(du_dt * v, dim=1)  
        left1 += self.M * torch.mean(torch.sum(dmu_dx * dv, dim=2, keepdims=True), dim=1)
        left2 = torch.mean( mu * v, dim=1) 
        left2 += - torch.mean( (u**3-u) * v, dim=1)
        left2 += - self._lambda2 * torch.mean(torch.sum(du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        # f = self.fun_f(x=x, t=t).view(-1, m, 1)
        # right = torch.mean(f * v, dim=1)

        return left1, left2
    
    # Intermediate function mu
    def weak_inverseF(self, model_u, model_E,x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
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
        m = x_scaled.shape[0] # m是球内的积分点
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
        du_d2x = self._Laplace_u([x], du_dx)
        E = E.view(-1, 1)
        #
        mu = E - self._lambda2 * du_d2x
        dmu_dx = self._grad_u(x, mu)
        u, mu, du_dt, dmu_dx = u.view(-1, m, 1), mu.view(-1, m, 1), du_dt.view(-1, m, 1), dmu_dx.view(-1, m, self.dim)
        E = E.view(-1, m, 1)
        du_dx = du_dx.view(-1, m, self.dim)      
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form
        left = torch.mean(du_dt * v, dim=1)  
        left += self.M * torch.mean(torch.sum(dmu_dx * dv, dim=2, keepdims=True), dim=1)
        left += torch.mean( mu * v, dim=1) 
        left += - torch.mean( E  * v, dim=1)
        left += - self._lambda2 * torch.mean(torch.sum(du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        f = self.fun_f(x=x, t=t).view(-1, m, 1)
        right = torch.mean(f * v, dim=1)

        return left-right

    def weak_mu_invF(self, model_u, model_mu, model_E, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
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
        m = x_scaled.shape[0] # m是球内的积分点
        x = x_scaled * R + xc  
        t = t.repeat(1,m,1)
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        u = model_u(torch.cat([t,x], dim=1))
        mu = model_mu(torch.cat([t,x], dim=1))
        E = model_E(u.detach().requires_grad_(True).to(device))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        du_d2x = self._Laplace_u([x], du_dx)
        #
        dmu_dx = self._grad_u(x, mu)
        u, mu, du_dt, dmu_dx = u.view(-1, m, 1), mu.view(-1, m, 1), du_dt.view(-1, m, 1), dmu_dx.view(-1, m, self.dim)
        #
        
        # mu_true = E - self._lambda2 * du_d2x
        # mu_true = mu_true.view(-1, m, 1)
        # left = torch.mean(mu - mu_true, dim=1) 
        
        #
        E = E.view(-1, m, 1)
        du_dx = du_dx.view(-1, m, self.dim)      
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form        
        left1 = torch.mean(du_dt * v, dim=1)  
        left1 += self.M * torch.mean(torch.sum(dmu_dx * dv, dim=2, keepdims=True), dim=1)
        
        left2 = torch.mean( mu * v, dim=1) 
        left2 += - torch.mean( E  * v, dim=1)
        left2 += - self._lambda2 * torch.mean(torch.sum(du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        # f = self.fun_f(x=x, t=t).view(-1, m, 1)
        # right = torch.mean(f * v, dim=1)

        return left1, left2