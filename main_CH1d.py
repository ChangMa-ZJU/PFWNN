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

    def get_error(self, loadfig_path, savefig_path, data_path, net_type: str, names: list, num: int = 1):
        '''
        '''
        l2_error_best = np.empty((len(names), num))
        time = np.empty((len(names), num))
        abs_error_best = np.empty((len(names), num))
        #
        times_list = []
        l2_errs_list = []
        loss_list = []
        x_test_list, abs_list = [], []
        point_wise_list = []
        u_pred_list = []
        for i in range(len(names)):
            times, l2_errs, loss = [], [], []
            x_test, abs = 0., 0.
            point_wise_err = 0.
            u_pred = 0.
            for j in range(num):
                save_path = loadfig_path
                loss_err_save = scipy.io.loadmat(save_path + 'loss_error_saved.mat')
                data_test = scipy.io.loadmat(save_path + 'test_saved.mat')
                #
                times.append(loss_err_save['time'])
                l2_errs.append(loss_err_save['error'])
                loss.append(loss_err_save['loss_u'])
                x_test += data_test['x_test_preall']
                abs += np.abs(data_test['u_pred_preall'] - data_test['u_test_preall'])

                #
                l2_error_best[i, j] = min(loss_err_save['error'][0])
                #
                abs_error_best[i, j] = np.max(
                    np.abs(data_test['u_pred_preall'][:, 0] - data_test['u_test_preall'][:, 0]))
                point_wise_err += np.abs(data_test['u_pred_preall'][:, 0] - data_test['u_test_preall'][:, 0])
                u_pred += data_test['u_pred_preall'][:, 0]
                #
                time[i, j] = loss_err_save['time'][0][-1]
            ####
            times_list.append(np.concatenate(times, axis=0))
            l2_errs_list.append(np.concatenate(l2_errs, axis=0))
            loss_list.append(np.concatenate(loss, axis=0))
            x_test_list.append(x_test / num)
            abs_list.append(abs / num)
            point_wise_list.append(point_wise_err / num)
            u_pred_list.append(u_pred / num)
        #
        self.t, self.x, self.u = data_test['t_test_preall'], data_test['x_test_preall'], data_test['u_test_preall'][:,
                                                                                         0]
        ##################
        print('abs err (avg):', np.mean(abs_error_best, axis=1),
              'abs err (std):', np.std(abs_error_best, axis=1))
        print('l2 err (avg):', np.mean(l2_error_best, axis=1),
              'l2 err (std):', np.std(l2_error_best, axis=1))
        print('Time (avg):', np.mean(time, axis=1),
              'Time (std):', np.std(time, axis=1))
        # Draw().MBP([self.t], [self.x], [u_pred_list],label_list=[r'Predicted $\phi_{{}_{NN}}$'],
        #            save_path=savefig_path + "CH1d_MBP.pdf")
        # ############### Relative error vs. time(s)
        # Draw().show_confid_time(times_list, l2_errs_list,
        #                         x_name=r'Time(s)', y_name=r'Relative error of $\phi$',
        #                         confid=True,
        #                         save_path=savefig_path + "AC1d_l2err_time.pdf")
        # ############### loss vs. time(s)
        # Draw().show_confid_time(times_list, loss_list,
        #                         x_name=r'Time(s)', y_name=r'Loss',
        #                         confid=True,
        #                         save_path=savefig_path + "AC1d_loss_time.pdf")
        # ############### point-wise error
        # Draw().show_tRely_1d_list([self.t] * 3, [self.x] * 3, [self.u, u_pred_list, point_wise_list[0]],
        #                           label_list=[r'Reference $\phi$', r'Predicted $\phi_{{}_{NN}}$',
        #                                       r'|$\phi_{{}_{NN}} -\phi$|'],
        #                           save_path=savefig_path + "AC1d_true_pred_pointwise.pdf")
        # ############## timestamp error
        # true_data = scipy.io.loadmat(data_path)
        # usol = true_data['uu'][:, :-1]
        # x_star = true_data['x'][0]
        # u_pred = np.array(u_pred_list).reshape(usol.shape)
        # Draw().plot_1d_timestamp(x_star, usol, u_pred, save_path=savefig_path + "timestamp_error.pdf")

        # log-log
        # Draw().loglog(loss_list, l2_errs_list, xlabel=r'Loss', ylabel=r'Relative error of $\phi$', save_path=savefig_path + "log_log_u.pdf")

    def solve(self, step, stage, solver):
        #
        solver.train()
        solver.test(step, stage)

    def WNN(self, t1, t2, step, stage, inx: int, name: str = 'PFWNN'):
        '''
        '''
        from Solvers.PFWNN_CH import PFWNN
        ############
        kwargs = {'Num_particles': 100,
                  'Num_tin_size': 30,
                  'Nx_integral': 10,
                  'Nt_integral': None,
                  'topk': 4000,
                  'train_xbd_size_each_face': 1,
                  'train_tbd_size': 200,
                  'train_init_size': 512,
                  #
                  'maxIter': 50000,
                  'lr': 1e-3,
                  'w_weak': 100.,
                  'tol': 0.1,
                  'w_bd': 5.,
                  'w_init': 100.,
                  'R_way': 'Rdescend',
                  'R_max': 5e-2,
                  'R_min': 1e-3,
                  'net_type': 'tanh_per',
                  'test_fun': 'Wendland',
                  'int_method': 'mesh',
                  'hidden_width': 128,
                  'hidden_layer': 4,
                  'dtype': {'numpy': self.np_type, 'torch': self.torch_type},
                  't1': t1,
                  't2': t2,
                  'stage': stage,
                  'step': step,
                  'interval': time_interval,
                  'loadpath': data_path,
                  #
                  'coe_M': 0.01,
                  'coe_lam2M': 0.0001,
                  #
                  'Nc_append': 50,
                  'ada_step': 1000
                  }
        kwargs['saved_path'] = current_directory + '/SavedModels/PFWNN/' + model_name + f"/{kwargs['stage']}/"
        #
        print("The parameters are:", kwargs)
        ##############
        solver = PFWNN(Problem=Problem, **kwargs)
        self.solve(step=kwargs['step'], stage=kwargs['stage'], solver=solver)
        np.save(kwargs['saved_path'] + 'settings.npy', kwargs)


if __name__ == '__main__':
    from Problems.CahnHilliard1d import Problem

    ############################################
    demo = Example(np_type=np.float32, torch_type=torch.float32)
    time_interval = int(1 / 0.005)
    time_step = 20
    # time intervals
    start = 0
    end = 1.
    num_stages = 10
    step = (end - start) / num_stages
    time_group = [[start + i * step, start + (i + 1) * step, i, time_step] for i in range(num_stages)]
    # time_group = [[0.9, 1.0, 9, 20]]
    #
    model_name = "CH_1d"
    loadfig_path = current_directory + '/SavedModels/PFWNN/' + model_name + f'/9/'
    savefig_path = current_directory + '/Results/PFWNN/' + model_name + '/'
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)
    data_path = current_directory + '/Data/cahn-hillard1d.mat'
    #
    # for time_stage in time_group:
    #     print('The time stage is:', time_stage)
    #     demo.WNN(t1=time_stage[0], t2=time_stage[1], stage=time_stage[2], step=time_stage[3], inx=0)
    ####
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)
    demo.get_error(loadfig_path, savefig_path, data_path, net_type='tanh_per', names=['PFWNN'], num=1)

