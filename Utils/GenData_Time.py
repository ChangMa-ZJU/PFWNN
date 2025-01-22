 # /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:07:31 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:07:31 
#  */
import numpy as np 
import torch 
import os 
import sys 
from scipy.stats import qmc
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#
from Problems.Module_Time import Problem

class GenData():
    '''
    '''
    def __init__(self, problem:Problem, dtype:np.dtype=np.float32):
        self.problem = problem
        self.dtype = dtype
        self.d = self.problem.dim
        self.x_lb = self.problem._lb
        self.x_ub = self.problem._ub
        self.lhs_t = qmc.LatinHypercube(1)
        self.lhs_x = qmc.LatinHypercube(self.problem.dim)
    
    def get_in(self, t0:float, tT:float, Nt_size:int, Nx_size:int,
               t_method:str='hypercube'):
        '''
        Input:
            t0, tT :
            Nx_size: meshsize for x-axis
            Nt_size: meshsize for t-axis
            t_method: 
        Return:
             x: size(Nx_size * Nt_size, d)
             t: size(Nx_size * Nt_size, 1)
        '''
        if t_method=='mesh':
            t = np.linspace(t0, tT, Nt_size).reshape(-1,1)  
        elif t_method=='random':
            t = np.random.uniform(t0, tT, [Nt_size,1])  
        elif t_method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), t0, tT)
        else:
            raise NotImplementedError
        #（size = Nx_size * Nt_size)
        x = qmc.scale(self.lhs_x.random(Nx_size*Nt_size), self.x_lb, self.x_ub)
        t = t.repeat(Nx_size, axis=0)
        
        return torch.tensor(x.astype(self.dtype)), torch.tensor(t.astype(self.dtype))
    
    def get_bd(self, t0:float, tT:float, Nt_size:int, Nx_bd_each_face:int, 
               t_method='hypercube'):
        '''
        Input:
                    t0, tT:
            Nx_bd_each_face: mesh-size in the x-axis
                   Nt_size: mesh-size in the t-axis
                   t_methd: 'mesh' or 'hypercube'
        Return:
             x_list : [ size(N_bd_each_face * Nt_size, 1) ] * 2d
                      where x_list has the form [lb_d1, ub_d1, lb_d2, ub_d2, ......]
             t: size(N_bd_each_face * Nt_size, 1)
        '''
        x_list = []
        if t_method=='mesh':
            t = np.linspace(t0, tT, Nt_size).reshape(-1,1)
        elif t_method=='random':
            t = np.random.uniform(t0, tT, [Nt_size,1])  
        elif t_method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), t0, tT)
        else:
            raise NotImplementedError
        # size = [N_bd_each_face * Nt_size] * 2 * d
        x = qmc.scale(self.lhs_x.random(Nx_bd_each_face * Nt_size), self.x_lb, self.x_ub)
        for d in range(self.d):
            x_lb, x_ub= np.copy(x), np.copy(x)
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self.x_lb[d], self.x_ub[d]
            # x_list.extend([torch.from_numpy(x_lb.astype(self.dtype)), 
            #                torch.from_numpy(x_ub.astype(self.dtype))])
        t = t.repeat(Nx_bd_each_face, axis=0)

        return torch.from_numpy(x_lb.astype(self.dtype)), torch.from_numpy(x_ub.astype(self.dtype)), torch.from_numpy(t.astype(self.dtype))

    def get_init(self, Nx_init:int)->torch.tensor:
        '''
        Input:   N_init: Number of points at initial time
                 given_t: The given time stamp
        Output:    x:size(?,d)
                   t:size(?,1)
        '''
        # (Nx_init)
        x = qmc.scale(self.lhs_x.random(Nx_init), self.problem._lb, self.problem._ub)
        t = 0. * np.ones([Nx_init, 1])

        return torch.from_numpy(x.astype(self.dtype)), torch.from_numpy(t.astype(self.dtype))
    
    def get_txc(self, t1, t2, N_xc:int, Nt_size:int, 
                R_max:float=1e-3, R_min:float=1e-8,
                R_method:str='R_first', t_method:str='mesh')->torch.tensor:
        '''
        Input: N_xc: particles
               Nt_size: 
               R_max: 
               R_min:
               R_method: 
               t_method: 
        Output: R, xc, t
        '''
        if R_max<R_min:
            raise ValueError('R_max should be large than R_min.')
        elif (2.*R_max)>np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        elif (R_max)<(1e-7+1e-8) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        elif (R_max)<(1e-15+1e-16) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        #
        R = np.random.uniform(R_min, R_max, [N_xc * Nt_size, 1])
        lb, ub = self.problem._lb + R, self.problem._ub - R
        #
        if R_method=='R_first':
            if t_method=='mesh':
                t = np.linspace(t1, t2, Nt_size).reshape(-1,1)
            elif t_method=='hypercube':
                t = qmc.scale(self.lhs_t.random(Nt_size), t1, t2)
            else:
                raise NotImplementedError
            # N_xc * Nt_mesh
            xc = self.lhs_x.random(N_xc*Nt_size) * (ub - lb) + lb 
            t = t.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError

        # Figure
        # x_min, x_max = 0, 1
        # y_min, y_max = 0, 1
        # points = self.lhs_x.random(N_xc) * (self.problem._ub - self.problem._lb) + self.problem._lb
        # # 定义每个点的半径
        # radii = np.random.uniform(R_min, R_max, [N_xc, 1])
        # fig, ax = plt.subplots()
        # # 绘制正方形区域
        # ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'k-')
        # # 为每个点绘制圆圈
        # for point, r in zip(points, radii):
        #     circle = plt.Circle(point, r, edgecolor='b', facecolor='none', lw=1, alpha=0.5)
        #     ax.add_artist(circle)
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        # # 设置坐标轴的比例为相等，以确保正方形不会变形
        # ax.set_aspect('equal')
        # plt.show()
        # plt.savefig('region.png')
        
        return torch.tensor(R.astype(self.dtype)).view(-1, 1, 1),\
            torch.tensor(xc.astype(self.dtype)).view(-1,1,self.problem.dim),\
                torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)

    def get_txc_adaptive(self, t1, t2, N_xc: int, Nt_size: int,num_add:int,
                R_max: float = 1e-3, R_min: float = 1e-8,
                R_method: str = 'R_first', t_method: str = 'mesh') -> torch.tensor:
        '''
        Input: N_xc: particles
               Nt_size:
               R_max:
               R_min:
               R_method:
               t_method:
        Output: R, xc, t
        '''
        if R_max < R_min:
            raise ValueError('R_max should be large than R_min.')
        elif (2. * R_max) > np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        elif (R_max) < (1e-7 + 1e-8) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        elif (R_max) < (1e-15 + 1e-16) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        #
        R = np.random.uniform(R_min, R_max, [N_xc * Nt_size +num_add, 1])
        R1 = np.random.uniform(R_min, R_max, [N_xc * Nt_size , 1])
        lb, ub = self.problem._lb + R1, self.problem._ub - R1
        #
        if R_method == 'R_first':
            if t_method == 'mesh':
                t = np.linspace(t1, t2, Nt_size).reshape(-1, 1)
            elif t_method == 'hypercube':
                t = qmc.scale(self.lhs_t.random(Nt_size), t1, t2)
            else:
                raise NotImplementedError
            # N_xc * Nt_mesh
            xc = self.lhs_x.random(N_xc * Nt_size) * (ub - lb) + lb
            t = t.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError

        return torch.tensor(R.astype(self.dtype)).view(-1, 1, 1), \
            torch.tensor(xc.astype(self.dtype)).view(-1, 1, self.problem.dim), \
            torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)

    def get_res_txc(self, t1, t2, N_xc: int, Nt_size: int,N_add,
                R_max: float = 1e-3, R_min: float = 1e-8,
                R_method: str = 'R_first', t_method: str = 'mesh') -> torch.tensor:
        '''
        Input: N_xc: particles
               Nt_size:
               R_max:
               R_min:
               R_method:
               t_method:
        Output: R, xc, t
        '''
        if R_max < R_min:
            raise ValueError('R_max should be large than R_min.')
        elif (2. * R_max) > np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        elif (R_max) < (1e-7 + 1e-8) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        elif (R_max) < (1e-15 + 1e-16) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        #
        R_all = np.random.uniform(R_min, R_max, [(N_xc+N_add) * Nt_size, 1])
        R = np.random.uniform(R_min, R_max, [N_xc * Nt_size, 1])
        lb, ub = self.problem._lb + R, self.problem._ub - R
        #
        if R_method == 'R_first':
            if t_method == 'mesh':
                t = np.linspace(t1, t2, Nt_size).reshape(-1, 1)
            elif t_method == 'hypercube':
                t = qmc.scale(self.lhs_t.random(Nt_size), t1, t2)
            else:
                raise NotImplementedError
            # N_xc * Nt_mesh
            xc = self.lhs_x.random(N_xc * Nt_size) * (ub - lb) + lb
            t = t.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError

        return torch.tensor(R_all.astype(self.dtype)).view(-1, 1, 1), \
            torch.tensor(xc.astype(self.dtype)).view(-1, 1, self.problem.dim), \
            torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)

    def get_x_scaled(self, Nx_scaled, method:str='mesh')->torch.tensor:
        '''
        Input: Nx_scaled: 
               method: 'mesh'
        Output: x_scaled
        '''
        if method=='mesh':
            if self.problem.dim==1:
                x_scaled = np.linspace(-1., 1., Nx_scaled).reshape(-1, self.problem.dim) # 这里必须是【-1，1】
            elif self.problem.dim==2:
                x, y = np.meshgrid(np.linspace(-1., 1., Nx_scaled), np.linspace(-1., 1., Nx_scaled))
                X_d = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0] # 半径是1
                x_scaled = X_d[index,:]
            else:
                raise NotImplementedError('The mesh method is not availabel for d>3.')
        else:
            raise NotImplementedError
            
        return torch.tensor(x_scaled.astype(self.dtype))


    def get_txc_adap(self, t1, t2,  Nt_size: int,N_xc,xc_append,
                     R_max: float = 1e-3, R_min: float = 1e-8,
                     t_method: str = 'mesh') -> torch.tensor:
        #
        N_xc_add = xc_append.shape[0]
        R = np.random.uniform(R_min, R_max, [(N_xc+N_xc_add) * Nt_size, 1])
        lb, ub = self.problem._lb + R, self.problem._ub - R
        #
        if t_method == 'mesh':
            t = np.linspace(t1, t2, Nt_size).reshape(-1, 1)
        elif t_method == 'hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), t1, t2)
        else:
            raise NotImplementedError
        # N_xc * Nt_mesh
        xc = self.lhs_x.random(N_xc * Nt_size)
        if N_xc_add != 0:
            xc_append = xc_append.repeat(Nt_size, axis=0)
            xc = np.concatenate((xc, xc_append), axis=0)* (ub - lb) + lb
        else:
            xc = xc * (ub - lb) + lb
        t = t.repeat((N_xc+N_xc_add), axis=0)

        return torch.tensor(R.astype(self.dtype)).view(-1, 1, 1), \
            torch.tensor(xc.astype(self.dtype)).view(-1, 1, self.problem.dim), \
            torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)

    def get_t_scaled(self, Nt_scaled, method:str='mesh')->torch.tensor:
        '''
        '''
        if method=='mesh':
            t_scaled = np.linspace(-1., 1., Nt_scaled).reshape(-1, 1)
        else:
            raise NotImplementedError
            
        return torch.tensor(t_scaled.astype(self.dtype))

    def get_txc_log(self, t1, t2, N_xc: int, Nt_size: int,
                R_max: float = 1e-3, R_min: float = 1e-8,
                R_method: str = 'R_first', t_method: str = 'mesh') -> torch.tensor:

        if R_max < R_min:
            raise ValueError('R_max should be large than R_min.')
        elif (2. * R_max) > np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        elif (R_max) < (1e-7 + 1e-8) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        elif (R_max) < (1e-15 + 1e-16) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        #
        R = np.random.uniform(R_min, R_max, [N_xc * Nt_size, 1])
        lb, ub = self.problem._lb + R, self.problem._ub - R
        #
        if R_method == 'R_first':
            if t_method == 'mesh':
                t = np.linspace(t1, t2, Nt_size).reshape(-1, 1)
            elif t_method == 'hypercube':
                t = qmc.scale(self.lhs_t.random(Nt_size), t1, t2)
            else:
                raise NotImplementedError
            # N_xc * Nt_mesh
            xc = self.lhs_x.random(N_xc * Nt_size) * (ub - lb) + lb
            t = t.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError

        R = torch.tensor(R.astype(self.dtype)).view(-1, 1, 1)
        xc = torch.tensor(xc.astype(self.dtype)).view(-1, 1, self.problem.dim)
        tc = torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)

        xc_list, tc_list, R_list = [xc], [tc], [R]
        ###############
        data_point = {'xc_list': xc_list, 'tc_list': tc_list, 'R_list': R_list}

        return data_point