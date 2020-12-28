import numpy as np
import json
from importlib import reload
import os
from models.core.tf_models.cae_model import CAE
import pickle
import tensorflow as tf
import dill
from collections import deque
from models.core.tf_models import utils
from scipy.interpolate import CubicSpline
import time
from models.core.tf_models import cae_model
reload(cae_model)
from models.core.tf_models.cae_model import CAE

class GenModel():
    """TODO:
    - reward (A*state + B*Belief)
    - can be attr of the mcts planner
    """
    def __init__(self):
        self.set_stateIndex()

    def set_stateIndex(self):
        self.indx_m = {}
        self.indx_y = {}
        self.indx_f = {}
        self.indx_fadj = {}
        i = 0
        for name in ['vel', 'pc', 'act_long_p','act_lat_p']:
            self.indx_m[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_y[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_f[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_fadj[name] = i
            i += 1

    def get_dx(self, vel_m, vel_o, veh_orientation):
        """
        veh_orientation is mering vehicle's orientation relative to others
        """
        if veh_orientation == 'front':
            dv = vel_m - vel_o
        else:
            dv = vel_o - vel_m

        dx = dv*0.1

        return dx

    def step(self, st_arr_i, acts_arr_i):
        """
        :Return: state_ii, state at the next time_step
        """
        st_arr_ii = st_arr_i.copy()
        st_arr_ii[:, self.indx_m['vel']] += acts_arr_i[:, 0]*0.1
        st_arr_ii[:, self.indx_y['vel']] += acts_arr_i[:, 2]*0.1
        st_arr_ii[:, self.indx_f['vel']] += acts_arr_i[:, 3]*0.1
        st_arr_ii[:, self.indx_fadj['vel']] += acts_arr_i[:, 4]*0.1

        indx = self.indx_m['pc']
        st_arr_ii[:, indx] += acts_arr_i[:, 1]*0.1
        lc_left = st_arr_ii[:, indx] > self.max_pc
        st_arr_ii[lc_left, indx] = self.min_pc
        lc_right = st_arr_ii[:, indx] < self.min_pc
        st_arr_ii[lc_right, indx] = self.max_pc

        vel_m = st_arr_i[:, self.indx_m['vel']]
        vel_y = st_arr_i[:, self.indx_y['vel']]
        vel_f = st_arr_i[:, self.indx_f['vel']]
        vel_fadj = st_arr_i[:, self.indx_fadj['vel']]
        st_arr_ii[:, self.indx_y['dx']] += self.get_dx(vel_m, vel_y, 'front')
        st_arr_ii[:, self.indx_f['dx']] += self.get_dx(vel_m, vel_f, 'behind')
        st_arr_ii[:, self.indx_fadj['dx']] += self.get_dx(vel_m, vel_fadj, 'behind')

        st_arr_ii[:, self.indx_m['act_long_p']] = acts_arr_i[:, 0]
        st_arr_ii[:, self.indx_m['act_lat_p']] = acts_arr_i[:, 1]
        st_arr_ii[:, self.indx_y['act_long_p']] = acts_arr_i[:, 2]
        st_arr_ii[:, self.indx_f['act_long_p']] = acts_arr_i[:, 3]
        st_arr_ii[:, self.indx_fadj['act_long_p']] = acts_arr_i[:, 4]

        return st_arr_ii

    def forwardSim(self, st_arr, acts_arr, steps_n):
        """
        Simulates the world forward for given number of steps
        :Params:
        - st_arr is unscaled state vector for time_step0
                            shape: ([traj_n, states])
        - acts_arr is unscaled action sequence for pred_horizon
                            shape:([traj_n, steps, all-actions])

        :Return: Predicted future states
        """
        state_predictions = np.zeros([st_arr.shape[0], steps_n, st_arr.shape[1]])
        state_predictions[:, 0, :] = st_arr
        state_i = st_arr.copy()
        for step in range(1, steps_n):
            st_arr_ii = self.step(state_i, acts_arr[:, step, :])
            state_predictions[:, step, :] = st_arr_ii
            state_i = st_arr_ii

        return state_predictions

class MergePolicy():
    def __init__(self, test_data, config):
        self.loadModel(config)
        self.data_obj = test_data.data_obj

        # TODO:
        # objective function/ evaluate function/ set execution time, which will
        # execute best ranked traj for a period of time.

    def loadModel(self, config):
        checkpoint_dir = './models/experiments/'+config['exp_id'] +'/model_dir'
        self.model = CAE(config, model_use='inference')
        Checkpoint = tf.train.Checkpoint(net=self.model)
        # Checkpoint.restore(tf.train.latest_checpoint(checkpoint_dir)).expect_partial()
        Checkpoint.restore(checkpoint_dir+'/ckpt-16')

        self.enc_model = self.model.enc_model
        self.dec_model = self.model.dec_model

    def get_cae_outputs(self, seq, traj_n, pred_h):
        """Output includes both samplred action sequences and their correspoinding
            distributions.
        :Param: [state_seq, cond_seq], traj_n, pred_h(s)
        """
        st_seq, cond_seq = seq
        # reshape to fit model
        st_seq.shape = (1, st_seq.shape[0], st_seq.shape[1])

        for n in range(5):
            cond_seq[n].shape = (1, 1, 1)
            cond_seq[n] = np.repeat(cond_seq[n], traj_n, axis=0)

        st_seq = np.repeat(st_seq, traj_n, axis=0)
        # get enc_h state
        enc_state = self.enc_model(st_seq)

        self.skip_n = 3 # done for a smoother trajectory
        self.step_len = round(self.skip_n*self.data_obj.step_size*0.1, 1) # [s]
        steps_n = int(np.ceil(np.ceil(pred_h/self.step_len)*self.step_len/ \
                                                    (self.data_obj.step_size*0.1)))

        self.dec_model.steps_n = steps_n
        self.dec_model.traj_n = traj_n

        sampled_actions, gmm_mlon, gmm_mlat = self.dec_model([cond_seq, enc_state])
        return sampled_actions, gmm_mlon, gmm_mlat

    def construct_policy(self, unscaled_acts, bc_der, traj_n, pred_h):
        bc_der = np.repeat([bc_der], traj_n, axis=0)

        # spline fitting
        traj_len = self.dec_model.steps_n*self.data_obj.step_size*0.1 # [s]
        trajectories = np.zeros([traj_n, int(traj_len*10)+1, 5])
        x_whole = np.arange(0, traj_len+0.1, self.step_len)
        x_snippet = np.arange(0, traj_len+0.1, 0.1)
        # t0 = time.time()

        for act_n in range(5):
            f = CubicSpline(x_whole, unscaled_acts[:,:,act_n],
                                bc_type=(
                                (1, bc_der[:, act_n]),
                                (2, [0]*traj_n)), axis=1)
            coefs = np.stack(f.c, axis=2)
            trajectories[:, :, act_n] = f(x_snippet)

        # print(time.time() - t0)
        return trajectories[:, 0:pred_h*10,:]

    def get_actions(self, seq, bc_der, traj_n, pred_h):
        """
        :Return: unscaled action array for all cars
        """
        sampled_actions, gmm_mlon, gmm_mlat = self.get_cae_outputs(seq, traj_n, pred_h)
        act_mlon, act_mlat, act_y, act_f, act_fadj = sampled_actions
        st_seq, cond_seq = seq

        total_acts_count = traj_n*self.dec_model.steps_n
        veh_acts_count = 5 # 2 for merging, 1 for each of the other cars
        scaled_acts = np.zeros([total_acts_count, veh_acts_count])
        i = 0
        actions = [act_.numpy() for act_ in [act_mlon, act_mlat, act_y, act_f, act_fadj]]
        for act_ in actions:
            act_.shape = (total_acts_count)
            scaled_acts[:, i] = act_

            i += 1

        unscaled_acts = self.data_obj.action_scaler.inverse_transform(scaled_acts)
        unscaled_acts.shape = (traj_n, self.dec_model.steps_n, veh_acts_count)

        cond0 = [cond_seq[n][0, 0, :].tolist() for n in range(5)]
        cond0 = np.array([item for sublist in cond0 for item in sublist])
        cond0 = self.data_obj.action_scaler.inverse_transform(np.reshape(cond0, [1,-1]))
        cond0.shape = (1, 1, 5)
        cond0 = np.repeat(cond0, traj_n, axis=0)
        unscaled_acts = np.concatenate([cond0, unscaled_acts], axis=1)

        return self.construct_policy(unscaled_acts[:,0::self.skip_n,:], bc_der, traj_n, pred_h)

class TestdataObj():
    dirName = './datasets/preprocessed/'
    def __init__(self, traffic_density, config):
        self.traffic_density = traffic_density
        self.setup(config['data_config']) # load test_data and validation data

    def setup(self, data_config):
        self.test_episodes = np.loadtxt('./datasets/'+self.traffic_density+\
                                    'test_episodes.csv', delimiter=',')

        config_names = os.listdir(self.dirName+'config_files')
        for config_name in config_names:
            with open(self.dirName+'config_files/'+config_name, 'r') as f:
                config = json.load(f)

            if config == data_config:
                with open(self.dirName+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = dill.load(f, ignore=True)

            if self.traffic_density == '':
                with open(self.dirName+config_name[:-5]+'/'+self.traffic_density+\
                                                            'states_test', 'rb') as f:
                    self.states_set = pickle.load(f)

                with open(self.dirName+config_name[:-5]+'/'+self.traffic_density+\
                                                            'targets_test', 'rb') as f:
                    self.targets_set = pickle.load(f)
            else:
                with open(self.dirName+self.traffic_density+'states_test', 'rb') as f:
                    self.states_set = pickle.load(f)

                with open(self.dirName+self.traffic_density+'targets_test', 'rb') as f:
                    self.targets_set = pickle.load(f)

class ModelEvaluation():
    def __init__(self, model, test_data, config):
        self.policy = model
        self.test_data = test_data
        self.gen_model = GenModel()
        self.episode_n = 50
        self.traj_n = 50
        self.pred_h = 2 # [s]
        self.dirName = './models/experiments/'+config['exp_id']
        data_obj = self.test_data.data_obj

    def obsSequence(self, state_arr, target_arr, test_data):
        state_arr = test_data.data_obj.applyStateScaler(state_arr)
        target_arr = test_data.data_obj.applyActionScaler(target_arr)
        actions = [target_arr[:, n:n+1] for n in range(5)]
        traj_len = len(state_arr)
        step_size = test_data.data_obj.step_size
        pred_step_n = int(np.ceil(self.pred_h/(step_size*0.1)))
        obs_n = test_data.data_obj.obs_n
        conds = [[],[],[],[],[]]
        states = []

        if traj_len > 20:
            prev_states = deque(maxlen=obs_n)
            for i in range(traj_len):
                prev_states.append(state_arr[i])

                if len(prev_states) == obs_n:
                    indx = np.arange(i, i+(pred_step_n+1)*step_size, step_size)
                    indx = indx[indx<traj_len]
                    if indx.size != pred_step_n+1:
                        break

                    states.append(np.array(prev_states))
                    for n in range(5):
                        conds[n].append(actions[n][indx[0:1]])

        return np.array(states),  [np.array(conds[n]) for n in range(5)]

    def episodeSetup(self, episode_id):
        """:Return: All info needed for evaluating model on a given scene
        """
        test_data = self.test_data
        st_arr = test_data.states_set[test_data.states_set[:, 0] == episode_id][:, 1:]
        targ_arr = test_data.targets_set[test_data.targets_set[:, 0] == episode_id][:, 1:]
        st_seq, cond_seq = self.obsSequence(st_arr.copy(), targ_arr.copy(), test_data)
        return st_seq, cond_seq, st_arr, targ_arr

    def sceneSetup(self, st_seq, cond_seq, st_arr, targ_arr, current_step, pred_h):
        """Set up a scence for a given initial step.
            Note: steps are index of numpy array, starting from 0.
        """
        start_step = current_step - 19
        end_step = int(current_step + pred_h/0.1)

        bc_der_i = (targ_arr[current_step, :]-targ_arr[current_step-1, :])*10
        st_seq_i = st_seq[start_step,:,:]

        cond_seq_i = [cond_seq[n][start_step,:,:] for n in range(5)]
        history_i = targ_arr[start_step:current_step+1, :]
        targ_i = targ_arr[current_step:end_step, :]
        st_i = st_arr[current_step:end_step, :]
        return st_seq_i, cond_seq_i, bc_der_i, history_i, st_i, targ_i

    def root_weightet_sqr(self, true_traj, pred_trajs):
        true_traj = true_traj[~np.all(true_traj == 0, axis=1)]
        pred_trajs = pred_trajs[~np.all(pred_trajs == 0, axis=1)]
        err = np.sqrt(np.mean(np.square(true_traj-pred_trajs), axis=0))
        return err

    def trajCompute(self, episode_id):
        st_seq, cond_seq, st_arr, targ_arr = self.episodeSetup(episode_id, self.test_data)
        actions = self.policy.get_actions([st_seq[29].copy(), cond_seq[29].copy()],
                                                    self.traj_n, self.pred_h)

        # simulate state forward
        state_i = np.repeat([st_arr[29, :]], self.traj_n, axis=0)
        self.gen_model.max_pc = max(st_arr[:, self.gen_model.indx_m['pc']])
        self.gen_model.min_pc = min(st_arr[:, self.gen_model.indx_m['pc']])
        state_predictions = self.gen_model.forwardSim(state_i.copy(), actions, self.pred_h)
        state_true = st_arr[0:29+self.pred_h]
        return state_true, state_predictions

    def compute_rwse(self, traffic_density):
        """
        dumps dict into exp folder containing RWSE for all vehicle actions across time.
        """
        # Ensure experiment has not beem done before

        file_names = os.listdir(self.dirName)
        if traffic_density+'rwse' in file_names:
            print("This experiment has been done already!")
            return None

        rwse_dict = {'vel_m':0,
                    'lat_vel':1,
                    'vel_y':2,
                    'vel_f':3,
                    'vel_fadj':4}

        pred_step_n = self.pred_h*10
        splits_n = 6 # number of splits across an entire trajectory
        pred_arrs = [np.zeros([self.episode_n*self.traj_n*6,
                                                pred_step_n]) for i in range(5)]
        truth_arrs = [np.zeros([self.episode_n*self.traj_n*6,
                                                pred_step_n]) for i in range(5)]
        _row = 0

        for episode_id in self.test_data.test_episodes[:self.episode_n]:
        # for episode_id in [1289]:
            st_seq, cond_seq, st_arr, targ_arr = self.episodeSetup(episode_id)
            self.gen_model.max_pc = max(st_arr[:, self.gen_model.indx_m['pc']])
            self.gen_model.min_pc = min(st_arr[:, self.gen_model.indx_m['pc']])
            if len(st_seq) >= 6:
                splits_n = 6
            else:
                splits_n = len(st_seq)

            traj_splits = np.random.choice(range(19, 19+len(st_seq)), splits_n, replace=False)

            for split in traj_splits:
                st_seq_i, cond_seq_i, bc_der_i, _, st_i, targ_i = self.sceneSetup(st_seq,
                                                                cond_seq,
                                                                st_arr,
                                                                targ_arr,
                                                                current_step=split,
                                                                pred_h=self.pred_h)
                targ_i.shape = (1, pred_step_n, 5)
                st_init = np.repeat(np.reshape(st_i[0,:], [1,17]), self.traj_n, axis=0)

                actions = self.policy.get_actions([st_seq_i, cond_seq_i], bc_der_i,
                                        traj_n=self.traj_n, pred_h=self.pred_h)
                st_pred = self.gen_model.forwardSim(st_init, actions, pred_step_n)

                truth_arrs[0][_row:_row+self.traj_n, :] = \
                                    st_i[:,self.gen_model.indx_m['vel']]
                pred_arrs[0][_row:_row+self.traj_n, :] = \
                                    st_pred[:,:,self.gen_model.indx_m['vel']]

                truth_arrs[1][_row:_row+self.traj_n, :] = \
                                    st_i[:,self.gen_model.indx_m['act_lat_p']]
                pred_arrs[1][_row:_row+self.traj_n, :] = \
                                    st_pred[:,:,self.gen_model.indx_m['act_lat_p']]

                truth_arrs[2][_row:_row+self.traj_n, :] = \
                                    st_i[:,self.gen_model.indx_y['vel']]
                pred_arrs[2][_row:_row+self.traj_n, :] = \
                                    st_pred[:,:,self.gen_model.indx_y['vel']]

                truth_arrs[3][_row:_row+self.traj_n, :] = \
                                    st_i[:,self.gen_model.indx_f['vel']]
                pred_arrs[3][_row:_row+self.traj_n, :] = \
                                    st_pred[:,:,self.gen_model.indx_f['vel']]

                truth_arrs[4][_row:_row+self.traj_n, :] = \
                                    st_i[:,self.gen_model.indx_fadj['vel']]
                pred_arrs[4][_row:_row+self.traj_n, :] = \
                                    st_pred[:,:,self.gen_model.indx_fadj['vel']]


                _row += self.traj_n
                # return st_pred
            print('Episode ', episode_id, ' has been completed!')
        for key in rwse_dict.keys():
            rwse_dict[key] = self.root_weightet_sqr(truth_arrs[rwse_dict[key]], \
                                                        pred_arrs[rwse_dict[key]])

        with open(self.dirName+'/'+ traffic_density + 'rwse', "wb") as f:
            pickle.dump(rwse_dict, f)
        return rwse_dict
