# /*
#  * @Author: chang.ma 
#  * @Date: 2023-12-09 14:32:32
#  * @Last Modified by: chang.ma 
#  * @Last Modified time: 2023-12-09 14:32:32 
#  */
import numpy as np
import torch
import scipy.io
from Utils.Draw import Draw
import os

current_directory = os.getcwd()

class Example():

    def __init__(self, np_type=np.float32, torch_type=torch.float32):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type


    def get_error(self, loadfig_path, savefig_path,data_path, net_type:str, names:list, num:int=1):
        '''
        '''
        l2_error_best = np.empty((len(names), num))
        time = np.empty((len(names), num))
        abs_error_best = np.empty((len(names), num))
        #
        times_list = []
        l2_errs_list = []
        E_errs_list = []
        loss_list = []
        x_test_list, abs_list = [], []
        point_wise_list = []
        u_pred_list = []
        for i in range(len(names)):
            times, l2_errs, E_errs, loss= [], [], [], []
            x_test, abs = 0., 0.
            point_wise_err = 0.
            u_pred = 0.
            for j in range(num):
                save_path = loadfig_path
                loss_err_save = scipy.io.loadmat(save_path+'loss_error_saved.mat')
                data_test = scipy.io.loadmat(save_path + 'test_saved.mat')
                #
                times.append(loss_err_save['time'])
                l2_errs.append(loss_err_save['error_u'])
                E_errs.append(loss_err_save['error_E'])
                loss.append(loss_err_save['loss'])
                x_test += data_test['x_test']
                abs += np.abs(data_test['u_pred'] - data_test['u_test'])
                #
                l2_error_best[i,j] =  min(loss_err_save['error_u'][0])
                #
                abs_error_best[i,j] = np.max(np.abs(data_test['u_pred'][:,0] - data_test['u_test'][:,0]))
                point_wise_err += np.abs(data_test['u_pred'][:,0] - data_test['u_test'][:,0])
                u_pred += data_test['u_pred'][:,0]
                #
                time[i,j] = loss_err_save['time'][0][-1]
            ####
            times_list.append(np.concatenate(times, axis=0))
            l2_errs_list.append(np.concatenate(l2_errs, axis=0))
            E_errs_list.append(np.concatenate(E_errs, axis=0))
            loss_list.append(np.concatenate(loss, axis=0))
            x_test_list.append(x_test/num)
            abs_list.append(abs/num)
            point_wise_list.append(point_wise_err/num)
            u_pred_list.append(u_pred/num)
        #
        self.t, self.x, self.u, self.u_pred= data_test['t_test'], data_test['x_test'], data_test['u_test'][:,0], data_test['u_pred']
        self.E_test, self.E_pred = data_test['E_test'], data_test['E_pred']
        err_E = np.mean( (self.E_pred - self.E_test)**2 ) / (np.mean(self.E_test**2))
        self.point_error_E = np.abs(self.E_test - self.E_pred)
        ##################
        print('abs err (avg):', np.mean(abs_error_best, axis=1), 
              'abs err (std):', np.std(abs_error_best, axis=1))
        print('l2 err (avg):', np.mean(l2_error_best, axis=1), 
              'l2 err (std):', np.std(l2_error_best, axis=1))
        print('Time (avg):', np.mean(time, axis=1), 
              'Time (std):', np.std(time, axis=1))
        print('error of E:', err_E)
        # ############### Relative error vs. time(s)
        # Draw().show_confid_time(times_list, l2_errs_list, 
        #                    x_name=r'Time(s)', y_name=r'Relative error of $\phi$',
        #                    confid=True,
        #                    save_path=savefig_path+'ACip1d_l2err_time.pdf')
        # ############### loss vs. time(s)
        # Draw().show_confid_time(times_list, loss_list,
        #                    x_name=r'Time(s)', y_name=r'Loss',
        #                    confid=True,
        #                    save_path=savefig_path+'ACips1d_loss_time.pdf')
        # ################ Relative error of energy E vs. time(s)
        # Draw().show_confid_time(times_list, E_errs_list, 
        #                    x_name='Time(s)', y_name=r'Relative error of energy',
        #                    confid=True,
        #                    save_path=savefig_path+"AC1dip_E_loss.pdf")
        # ############### energy function
        # Draw().show_energyfun([self.u, self.u_pred], [self.E_test, self.E_pred], 
        #                         [r'f$_{{}_{Reference}}$', r'f$_{{}_{Prediction}}$'],
        #                         x_name=r'$\phi$', y_name=r'f($\phi$)',
        #                         save_path=savefig_path+"AC1dip_contrast.pdf")
        # ############## point-wise error
        # Draw().show_tRely_1d_list([self.t]*3, [self.x]*3, [self.u, u_pred_list, point_wise_list[0]],
        #                           label_list=[r'Reference $\phi$', r'Predicted $\phi^{NN}$',
        #                                       r'|$\phi_{{}_{NN}} -\phi$|'],
        #                           save_path= savefig_path+ 'ACfp1d_pointwise.pdf')
        #  # ############## groundtruth-predict of energy
        # Draw().show_tRely_1d_list([self.t]*3, [self.x]*3, [self.E_test, self.E_pred, self.point_error_E],
        #                           label_list=[r'Reference $f(\phi)$', r'Predicted $f^{NN}(\phi^{NN})$',
        #                                       r'|$f^{NN}(\phi^{NN}) -f(\phi)$|'],
        #                           save_path=savefig_path+ 'ACip1d_energy_pointwise.pdf')
        # log-log
        Draw().loglog(loss_list, E_errs_list, xlabel=r'Loss', ylabel=r'Relative error of $f$', save_path=savefig_path + "log_log_f.pdf")
        # ############### timestamp error
        # true_data = scipy.io.loadmat(data_path)
        # usol = true_data['uu']
        # x_star = true_data['x'][0]
        # u_pred = np.array(u_pred_list).reshape(usol.shape)
        # Draw().plot_1d_timestamp(x_star, usol, u_pred, save_path= savefig_path+ 'timestamp_error.pdf')

        return err_E

       
        

    def solve(self, solver, save_path:str):
        '''
        '''
        solver.train(save_path)
        solver.test(save_path)

    def WNN(self,noise,inx:int, name:str='PFWNN'):
        '''
        '''
        from Solvers.PFWNN_AC_inverseF import PFWNN
        ############ inverse problem
        kwargs = {'Num_particles': 50,
                  'Num_tin_size': 100,
                  'Nx_integral': 10,
                  'Nt_integral': None,
                  'topk': 1500,
                  'train_xbd_size_each_face': 1,
                  'train_tbd_size': 0,
                  'train_init_size': 512,
                  #
                  'maxIter': 50000,  ###
                  'lr': 1e-3,
                  'w_weak': 100.,
                  'w_bd': 5.,
                  'w_init': 50.,
                  'w_data': 50.,
                  'tol': 1e-1, 
                  'R_way': 'Rdescend',
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  #
                  'net_type_u': 'tanh_per',
                  'net_type_E': 'tanh_sin',
                  'test_fun': 'Wendland',
                  'int_method': 'mesh', 
                  #
                  'hidden_width': 50,
                  'hidden_layer':3,
                  'hidden_width_E': 20,
                  'hidden_layer_E': 2,
                  'noise_level': noise,
                  'dtype':{'numpy':self.np_type, 'torch':self.torch_type},
                  #
                  'coe_M':5.,
                  'coe_lam2M':0.0001,
                  'data_path':data_path,
                  #
                  # 'portion_x':port_x,
                  # 'portion_t':port_t
                  #
                  't1': t1,
                  't2': t2,
                  'interval': time_interval
                  }
        #
        save_path = current_directory + '/SavedModels/PFWNN_IP/' + model_name + "/"
        #
        print(kwargs)
        ##############
        solver = PFWNN(Problem=Problem, **kwargs)
        self.solve(solver=solver, save_path=save_path)
        np.save(save_path+'settings.npy', kwargs)

if __name__=='__main__':
    from Problems.AllenCahn1d import Problem
    ############################################
    demo = Example(np_type=np.float32, torch_type=torch.float32)
    t1 = 0.
    t2 = 1.
    time_interval = int(1 / 0.005)
    noise_group = [0.05, 0.1, 0.15, 0.2]
    # portion_group = [[2, 2], [2, 5], [2, 10], [2, 40], [4, 2], [4, 5], [4, 10], [4, 40], [8, 2], [8, 5], [8, 10], [8, 40], \
    #     [16, 2], [16, 5], [16, 10], [16, 40]]
    errall = []
    for noise in noise_group:
        print('The noise is:', noise)
        model_name = "ACIP_1d"+f"_{noise}"''
        loadfig_path = current_directory + '/SavedModels/PFWNN_IP/' + model_name+"/"
        savefig_path = current_directory + '/Results/PFWNN_IP/' + model_name+"/"
        data_path = current_directory + '/Data/allen-cahn1d.mat'
        # demo.WNN(noise=noise,inx =0)
        if not os.path.exists(savefig_path):
            os.makedirs(savefig_path)
        err = demo.get_error(loadfig_path=loadfig_path, savefig_path=savefig_path, data_path =data_path, net_type='tanh_per', names=['PFWNN'], num=1)
        errall.append(err)
        print('#########',errall)


