import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
import json
from models.core.tf_models import utils
import matplotlib.pyplot as plt
from importlib import reload
import os
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import models.core.tf_models.abstract_model as am
import pickle

# %%

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

class YieldModel():
    def __init__(self, data_obj, config):
        # self.loadModel(config)
        self.data_obj = data_obj

    def get_actions(self, samples_n, time_step):

        return  actions.reshape(actions.shape[1], actions.shape[-1])

class MergePolicy():
    def __init__(self, data_obj, config):
        self.loadModel(config)
        self.data_obj = data_obj
        traj_n = 10

    def loadModel(self, config):
        checkpoint_dir = './models/experiments/'+config['exp_id'] +'/model_dir'
        self.model = am.FFMDN(config)
        Checkpoint = tf.train.Checkpoint(model=model)
        Checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def get_actions(self, state_arr):
        """
        :Param: Unscaled state vector
        :Return: Control actions for the current state_arr
        """
        state_arr = self.data_obj.applystateScaler(state_arr.reshape(-1, state_arr.shape[-1]))
        parameter_vector = self.model.predict(state_arr)
        alphas, mus_long, sigmas_long, mus_lat, sigmas_lat, rhos = np.split(parameter_vector[0], 6, axis=0)

        action_samples = utils.get_pdf_samples(1, parameter_vector, config['model_type'])
        actions = self.data_obj.applyInvScaler(action_samples)

        return  actions.reshape(actions.shape[1], actions.shape[-1])

class ModelEvaluation():
    dirName = './datasets/preprocessed/'

    def __init__(self, config):
        self.data_config = config['data_config']
        self.setup() # load data_obj and validation data
        self.policy = MergePolicy(self.data_obj, config)
        self.gen = GenModel(self.data_obj)
        # self.driver_model = YieldModel(data_obj, config)
        self.steps_n = 50 # time-steps into the future
        self.samples_n = 10 # time-steps into the future
        self.scalar_indx = self.data_obj.scalar_indx

        self.sceneSetup(episode_id=1635)

    def setup(self):
        config_names = os.listdir(self.dirName+'config_files')
        for config_name in config_names:
            with open(self.dirName+'config_files/'+config_name, 'r') as f:
                config = json.load(f)

            if config == self.data_config:
                with open(self.dirName+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = pickle.load(f)
                with open(self.dirName+config_name[:-5]+'/'+'val_m_df', 'rb') as f:
                    self.val_m_df = pickle.load(f)
                with open(self.dirName+config_name[:-5]+'/'+'val_y_df', 'rb') as f:
                    self.val_y_df = pickle.load(f)

    def trajCompute(self):
        """
        :Param:
            state0 - current vehicle state
        :Return: [[x_cor, y_cor], ...]
        """
        mveh_a = np.zeros([self.samples_n, self.steps_n-1, 2])
        state_arr = np.zeros([self.samples_n, self.steps_n, len(self.state_t0)])
        state_arr[:,0,:] = self.state_t0

        x = np.zeros([self.samples_n, self.steps_n])
        y = np.zeros([self.samples_n, self.steps_n])

        for t in range(1, self.steps_n):
            mveh_a[:,t-1,:] = self.policy.get_actions(state_arr[:,t-1,:].copy())
            yveh_a = np.repeat(self.yveh_as[t-1], self.samples_n, axis=0)
            x[:, t] = x[:, t-1] + state_arr[:,t-1,self.scalar_indx['vel_mveh']]*0.1
            y[:, t] = y[:, t-1] + mveh_a[:,t-1,1]*0.1
            state_arr[:,t,:] = self.gen.step(state_arr[:,t-1,:].copy(), mveh_a[:,t-1,:], yveh_a)

        return x, y, state_arr, mveh_a

    def sceneSetup(self, episode_id):
        m_df, y_df = self.data_obj.get_episode_df(self.val_m_df, self.val_y_df, episode_id)
        v_x_arr, v_y_arr = self.data_obj.get_stateTarget_arr(m_df, y_df)
        self.state_arr_true = np.array(v_x_arr[0:self.steps_n])
        v_x_arr, v_y_arr = self.data_obj.obsSequence(v_x_arr, v_y_arr)
        self.state_t0 = v_x_arr[0]
        self.yveh_as = y_df['act_long'].values

        self.gen.min_lane_width = m_df['pc'].min()
        self.gen.max_lane_width = m_df['pc'].max()
        x = np.zeros(self.steps_n)
        y = np.zeros(self.steps_n)
        vel = m_df['vel'].values
        act_lat = m_df['act_lat'].values
        for t in range(1, self.steps_n):
            x[t] = x[t-1] + vel[t]*0.1
            y[t] = y[t-1] + act_lat[t]*0.1
        self.x_true = x
        self.y_true = y
        self.mveh_a_true = m_df[['act_long', 'act_lat']].values[0:self.steps_n-1]

config = loadConfig('series000exp009')
eval = ModelEvaluation(config)
x, y, state_arr, mveh_a = eval.trajCompute()
plt.plot(eval.x_true, eval.y_true, color='red')
for i in range(10):
    plt.plot(x[i,:], y[i,:], color='grey')
# %%
eval.scalar_indx
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


# %%
max(eval.pc_true[0:90])
plt.plot(pc[0])
plt.plot(eval.pc_true[0:90])
plt.grid()
            pc[:, t] = state_arr[:, self.scalar_indx['pc_mveh']]

# %%

# %%

a = np.array([1,2,3,4,5,6])
np.split(a, 2, axis=0)

with open('./datasets/preprocessed/'+'20200924-153215'+'/'+'data_obj', 'rb') as f:
    data_obj = pickle.load(f)

config = loadConfig('exp001')
eval = ModelEvaluation(data_obj, config)
x, y = eval.trajCompute()
for i in range(10):
    plt.plot(x[i,:], y[i,:], color='grey')
x
# %%
controller = MergePolicy(data_obj, config)

data_obj.validation_episodes[3]
m_df, y_df = data_obj.get_episode_df(data_obj.val_m_df, data_obj.val_y_df, 1635)
v_x_arr, v_y_arr = data_obj.get_stateTarget_arr(m_df, y_df)
v_x_arr, v_y_arr = data_obj.obsSequence(v_x_arr, v_y_arr)
controller.get_actions(v_x_arr[0])

# %%

controller = MergePolicy(config)

m_df, y_df = data_prep.get_episode_df(811)
x_arr, y_arr = data_prep.get_stateTarget_arr(m_df, y_df)

# %%

x_arr[0]
controller.get_actions(x_arr[0:5], config)
a
a[:,0]
a.reshape(1, -1)
# %%
x, y = controller.trajCompute(controller.veh_state[0], config)
indx = controller.mvehindx

def true_traj():
    x = [0]
    y = [0]
    for i in range(10):
        x.append(x[-1] + controller.veh_state[i][indx['vel']]*0.1)
        y.append(x[-1] + controller.veh_state[i][indx['pc']]*0.1)

    return x, y
x_true, y_true = true_traj()
# %%
for i in range(10):
    plt.plot(np.array(x)[:,i], np.array(y)[:,i], color='black')

# %%
np.array(x)[:,0]


a = controller.veh_state[0:5]
a
from importlib import reload
reload(utils)





state_arr[0,:] = 0
trajCollection = []
for step in range(trajSteps_n):
