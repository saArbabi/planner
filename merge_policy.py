import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
import json
from models.core.tf_models import utils
import matplotlib.pyplot as plt
from importlib import reload
import os
from models.core.tf_models.cae_model import CAE
import pickle
import tensorflow as tf
import dill

# %%
class GenModel():
    """TODO:
    - reward (A*state + B*Belief)
    - can be attr of the mcts planner
    """
    def __init__(self, data_obj):
        self.step_size = 0.1
        self.scalar_indx = data_obj.scalar_indx
        self.retain_indx = data_obj.retain_indx
        self.m_s = data_obj.m_s
        self.y_s = data_obj.y_s

    def step(self, state_arr, mveh_a, yveh_a):
        indx = self.retain_indx['vel_mveh']
        state_arr[:,indx] += mveh_a[:,0]*self.step_size

        indx = self.retain_indx['vel_yveh']
        state_arr[:,indx] += yveh_a*self.step_size

        for state_key in self.m_s:
            indx = self.scalar_indx[state_key+'_mveh']
            if state_key == 'vel':
                state_arr[:,indx] += mveh_a[:,0]*self.step_size
            if state_key == 'pc':

                state_arr[:, indx] += mveh_a[:,1]*self.step_size
                lc_left = state_arr[:, indx]  > self.max_lane_width
                state_arr[lc_left, indx] = self.min_lane_width
                lc_right = state_arr[:, indx]  < self.min_lane_width
                state_arr[lc_right, indx] = self.max_lane_width

        for state_key in self.y_s:
            indx = self.scalar_indx[state_key+'_yveh']
            if state_key == 'vel':
                state_arr[:,indx] += yveh_a*self.step_size

            if state_key == 'dv':
                state_arr[:,indx] = state_arr[:, self.retain_indx['vel_mveh']] - \
                                        state_arr[:, self.retain_indx['vel_yveh']]

            if state_key == 'dx':
                state_arr[:,indx] += state_arr[:, self.scalar_indx['dv_yveh']]*self.step_size

            if state_key == 'da':
                state_arr[:,indx] = mveh_a[:, 0] - yveh_a

            if state_key == 'a_ratio':
                state_arr[:,indx] = np.log(np.abs(mveh_a[:, 0])/ np.abs(yveh_a))

        return state_arr

class MergePolicy():
    def __init__(self, data_obj, config):
        self.loadModel(config)
        self.data_obj = data_obj
        self.pred_horizon = 20
        # TODO:
        # objective function/ evaluate function/ set execution time, which will
        # execute best ranked traj for a period of time.

    def loadModel(self, config):
        checkpoint_dir = './models/experiments/'+config['exp_id'] +'/model_dir'
        self.model = CAE(config)
        Checkpoint = tf.train.Checkpoint(net=self.model)
        # Checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        Checkpoint.restore(checkpoint_dir+'/ckpt-7')

        self.enc_model = self.model.enc_model
        self.dec_model = self.model.dec_model

    def set_stepCondition(self, mveh_a, yveh_a):
        """
        :Return: Conditioning feature for each step.
        """
        condition_size = 2
        condition = np.zeros((1, 1, condition_size))
        condition[0, 0, 0] = mveh_a - yveh_a # da
        condition[0, 0, 1] = np.log(np.abs(mveh_a)/ np.abs(yveh_a))
        return condition

    def get_actions(self, gmm_m, gmm_y, samples_n):
        """
        :Param:
        - mixtures for both vehicles
        - number of samples
        :Return: sampled actions (scaled) for mveh and yveh
        """
        mveh_a = gmm_m.sample(samples_n).numpy()
        yveh_a = gmm_y.sample(samples_n).numpy()
        return np.reshape(mveh_a, [samples_n, 2]), np.reshape(yveh_a, [samples_n, 1])

    def trajCompute(self, obs_seq, condition_t0):
        """
        :Param:
        - sequence of obs
        - initial condition
        :Return: sequence of actions for mveh and yveh
        """
        mveh_traj = []
        yveh_traj = []

        enc_state = self.model.enc_model(obs_seq)
        condition = condition_t0[0]
        self.dec_model.state = enc_state
        for step in range(self.pred_horizon):
            condition.shape = (1, 1, 2)
            gmm_m, gmm_y = self.model.dec_model([condition ,self.dec_model.state])
            mveh_a, yveh_a = self.get_actions(gmm_m, gmm_y, 1)
            mveh_traj.append(mveh_a)
            yveh_traj.append(yveh_a)
            if step != (self.pred_horizon - 1):
                condition = self.set_stepCondition(mveh_a[0][0], yveh_a[0][0])
                # condition = condition_t0[step+1]

        mveh_traj = np.array(mveh_traj)
        yveh_traj = np.array(yveh_traj)
        mveh_traj.shape = (20, 2)
        yveh_traj.shape = (20, 1)
        mveh_traj = self.data_obj.apply_InvScaler(mveh_traj)
        yveh_traj = self.data_obj.apply_InvScaler(yveh_traj)

        return mveh_traj, yveh_traj

    # def trajCompute(self):
    #     """
    #     :Param:
    #         state0 - current vehicle state
    #     :Return: [[x_cor, y_cor], ...]
    #     """
    #     mveh_a = np.zeros([self.samples_n, self.steps_n-1, 2])
    #     state_arr = np.zeros([self.samples_n, self.steps_n, len(self.state_t0)])
    #     state_arr[:,0,:] = self.state_t0.copy()
    #     x = np.zeros([self.samples_n, self.steps_n])
    #     y = np.zeros([self.samples_n, self.steps_n])
    #     self.policy.time_stamp = 0
    #
    #     for t in range(1, self.steps_n):
    #         self.policy.ffadj_arr_t = np.repeat([self.ffadj_arr[t-1]], self.samples_n, axis=0)
    #         # self.policy.ffadj_arr_t = np.repeat([self.ffadj_arr[0]], self.samples_n, axis=0)
    #         mveh_a[:,t-1,:] = self.policy.get_actions(state_arr[:,t-1,:].copy())
    #         yveh_a = np.repeat(self.yveh_as[t-1], self.samples_n, axis=0)
    #         x[:, t] = x[:, t-1] + state_arr[:,t-1,self.scalar_indx['vel_mveh']]*0.1
    #         y[:, t] = y[:, t-1] + mveh_a[:,t-1,1]*0.1
    #         state_arr[:,t,:] = self.gen.step(state_arr[:,t-1,:].copy(), mveh_a[:,t-1,:], yveh_a)
    #         self.policy.time_stamp += 0.1
    #     return x, y, state_arr, mveh_a


    # def get_actions(self, state_arr):
    #     """
    #     :Param: Unscaled state vector
    #     :Return: Control actions for the current state_arr
    #     """
    #     state_arr = self.data_obj.applystateScaler(state_arr.reshape(-1, state_arr.shape[-1]))
    #     # param_vec = self.model(state_arr, training=False)
    #     param_vec = self.model(np.append(state_arr, self.ffadj_arr_t, axis=1), training=False)
    #     alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos = np.split(param_vec[0], 6, axis=0)
    #
    #     action_samples = utils.get_pdf_samples(1, param_vec, config['model_type'])
    #     actions = self.data_obj.applyInvScaler(action_samples)
    #
    #     return  actions.reshape(actions.shape[1], actions.shape[-1])

class ModelEvaluation():
    dirName = './datasets/preprocessed/'

    def __init__(self, config):
        self.data_config = config['data_config']
        self.setup() # load data_obj and validation data
        self.policy = MergePolicy(self.data_obj, config)
        # self.driver_model = YieldModel(data_obj, config)
        self.steps_n = 50 # time-steps into the future
        self.samples_n = 10 # time-steps into the future
        self.scalar_indx = self.data_obj.scalar_indx

        self.sceneSetup(episode_id=516)

    def setup(self):
        config_names = os.listdir(self.dirName+'config_files')
        for config_name in config_names:
            with open(self.dirName+'config_files/'+config_name, 'r') as f:
                config = json.load(f)

            if config == self.data_config:
                with open(self.dirName+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = dill.load(f, ignore=True)

                with open(self.dirName+config_name[:-5]+'/'+'test_m_df', 'rb') as f:
                    self.test_m_df = pickle.load(f)
                with open(self.dirName+config_name[:-5]+'/'+'test_y_df', 'rb') as f:
                    self.test_y_df = pickle.load(f)
                with open(self.dirName+config_name[:-5]+'/'+'test_ffadj_arr', 'rb') as f:
                    self.test_ffadj_arr = pickle.load(f)

    # def trajCompute(self):
    #     """
    #     :Param:
    #         state0 - current vehicle state
    #     :Return: [[x_cor, y_cor], ...]
    #     """
    #     mveh_a = np.zeros([self.samples_n, self.steps_n-1, 2])
    #     state_arr = np.zeros([self.samples_n, self.steps_n, len(self.state_t0)])
    #     state_arr[:,0,:] = self.state_t0.copy()
    #     x = np.zeros([self.samples_n, self.steps_n])
    #     y = np.zeros([self.samples_n, self.steps_n])
    #     self.policy.time_stamp = 0
    #
    #     for t in range(1, self.steps_n):
    #         self.policy.ffadj_arr_t = np.repeat([self.ffadj_arr[t-1]], self.samples_n, axis=0)
    #         # self.policy.ffadj_arr_t = np.repeat([self.ffadj_arr[0]], self.samples_n, axis=0)
    #         mveh_a[:,t-1,:] = self.policy.get_actions(state_arr[:,t-1,:].copy())
    #         yveh_a = np.repeat(self.yveh_as[t-1], self.samples_n, axis=0)
    #         x[:, t] = x[:, t-1] + state_arr[:,t-1,self.scalar_indx['vel_mveh']]*0.1
    #         y[:, t] = y[:, t-1] + mveh_a[:,t-1,1]*0.1
    #         state_arr[:,t,:] = self.gen.step(state_arr[:,t-1,:].copy(), mveh_a[:,t-1,:], yveh_a)
    #         self.policy.time_stamp += 0.1
    #     return x, y, state_arr, mveh_a

    def sceneSetup(self, episode_id):
        m_df, y_df = self.data_obj.get_episode_df(self.test_m_df, self.test_y_df, episode_id)
        self.m_id_df = m_df
        self.y_id_df = y_df

        state_arr, target_m_arr, target_y_arr, condition_arr = self.data_obj.get_stateTarget_arr(m_df, y_df)
        state_arr = self.data_obj.applystateScaler(state_arr)
        target_m_arr = self.data_obj.applytarget_mScaler(target_m_arr)
        target_y_arr = self.data_obj.applytarget_yScaler(target_y_arr)
        condition_arr = self.data_obj.applycondition_Scaler(condition_arr)
        f_x_arr = self.data_obj.get_ffadjState(self.test_ffadj_arr, episode_id)
        state_arr = np.concatenate([state_arr, f_x_arr], axis=1)

        self.state_arr, self.target_m_arr, _,  self.condition_arr = self.data_obj.obsSequence(
                                    state_arr, target_m_arr, target_y_arr,  condition_arr)

    def trajCompute(self):
        obs_seq = self.state_arr[0]
        obs_seq.shape = (1, 20, 15)
        condition_t0 = np.array(self.condition_arr[0])
        mveh_traj, yveh_traj = self.policy.trajCompute(obs_seq, condition_t0)
        self.st = obs_seq
        self.con = condition_t0
        # self.state_arr_true = np.array(v_x_arr[0:self.steps_n])
        # # v_x_arr, v_y_arr = self.data_obj.obsSequence(v_x_arr, v_y_arr)
        # self.ffadj_arr = self.data_obj.get_ffadjState(self.test_ffadj_arr, episode_id)
        # self.state_t0 = v_x_arr[0]
        # self.yveh_as = y_df['act_long'].values
        #
        # self.gen.min_lane_width = m_df['pc'].min()
        # self.gen.max_lane_widt h = m_df['pc'].max()
        # x = np.zeros(self.steps_n)
        # y = np.zeros(self.steps_n)
        # vel = m_df['vel'].values
        # act_lat = m_df['act_lat'].values
        # for t in range(1, self.steps_n):
        #     x[t] = x[t-1] + vel[t]*0.1
        #     y[t] = y[t-1] + act_lat[t]*0.1
        # self.x_true = x
        # self.y_true = y
        # self.mveh_a_true = m_df[['act_long', 'act_lat']].values[0:self.steps_n-1]

        return mveh_traj, yveh_traj

config = loadConfig('series002exp006')
eval_obj = ModelEvaluation(config)
plt.plot(eval_obj.m_id_df['act_long'])
# plt.plot(eval_obj.m_id_df['act_lat'])

# %%
# %%
fig, axs = plt.subplots(2, 1)
for i in range (10):
    mveh_traj, yveh_traj = eval_obj.trajCompute()
    axs[0].plot(range(19, 39), mveh_traj[:, 0])
    axs[1].plot(range(19, 39), yveh_traj)

axs[1].set_xlabel('time')
axs[0].set_ylabel('mveh_act_long')
axs[1].set_ylabel('yveh_act_long')
axs[0].grid(True)
axs[1].grid(True)

axs[0].plot(eval_obj.m_id_df['act_long'][0:40].values, color='red')
axs[1].plot(eval_obj.y_id_df['act_long'][0:40].values, color='red')

fig.tight_layout()
plt.show()


# %%
step = 0
obs_seq = eval_obj.state_arr[step]
obs_seq.shape = (1, 20, 15)
condition_t0 = np.array(eval_obj.condition_arr[step])
eval_obj.st = obs_seq
eval_obj.con = condition_t0
eval_obj.st[0][-1]
eval_obj.con.shape
condition_t0[0][0]
eval_obj.con.shape = (1, 20, 2)
gmm_m, gmm_y = eval_obj.policy.model([eval_obj.st, eval_obj.con])
for i in range (10):
    ma = gmm_m.sample().numpy()
    ma.shape = (20, 2)

    plt.plot(ma[:, 0], color='grey')
true = np.array(eval_obj.target_m_arr[step])[:, 0]
plt.plot(true, color='red')
plt.title('act_long')
plt.show()
# %%


# %%
step = 0
obs_seq = eval_obj.state_arr[step]
obs_seq.shape = (1, 20, 15)
condition_t0 = np.array(eval_obj.condition_arr[step])
eval_obj.st = obs_seq
eval_obj.con = condition_t0

eval_obj.con.shape = (1, 20, 4)
gmm_m, gmm_y = eval_obj.policy.model([eval_obj.st, eval_obj.con])
for i in range (10):
    ma = gmm_m.sample().numpy()
    ma.shape = (20, 2)

    plt.plot(ma[:, 1], color='grey')
true = np.array(eval_obj.target_m_arr[step])[:, 1]
plt.plot(true, color='red')
plt.show()


# %%
 tf.compat.v1.trainable_variables(
    scope=None)
# %%
plt.plot(mveh_traj[:, 1])

# %%
plt.plot(mveh_traj[:, 0])
plt.plot(eval_obj.test_m_df['act_long'][20:40].values)


# %%
        self.enc_model = self.model.enc_model
        self.dec_model = self.model.dec_model

eval_obj.policy.enc_model.trainable_weights[0]
eval_obj.policy.model.trainable_weights[0]
eval_obj.policy.dec_model.trainable_weights
# %%
"""State plots
"""
for item in eval.scalar_indx:
    fig = plt.figure()
    plt.plot(eval.state_arr_true[:, eval.scalar_indx[item]], color='red')
    for s in range(eval.samples_n):
        plt.plot(state_arr[s, :, eval.scalar_indx[item]], color='grey')
    plt.grid()
    plt.title(item)
    plt.xlabel('time step (n)')
    plt.xticks(range(eval.steps_n))

# %%
"""Action plots
"""
for item in ['act_long', 'act_lat']:
    if item == 'act_long':
        indx = 0
    else:
        indx = 1

    fig = plt.figure()
    plt.plot(eval.mveh_a_true[:, indx], color='red')
    for s in range(eval.samples_n):
        plt.plot(mveh_a[s, :, indx], color='grey')
    plt.grid()
    plt.title(item)
    plt.xlabel('time step (n)')
    plt.xticks(range(eval.steps_n))
