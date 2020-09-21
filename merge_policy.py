import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
from models.core.preprocessing import data_prep
DataPrep = data_prep.DataPrep
import json
from models.core.tf_models import utils
import matplotlib.pyplot as plt
from importlib import reload
import os
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
# %%

# %%
reload(data_prep)

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
                state_arr[:,indx] = np.log(np.abs(mveh_a[:, 0]))/ np.abs(yveh_a)

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
        dirName = './models/experiments/'+config['exp_id'] +'/trained_model'
        self.model = keras.models.load_model(dirName,
                                    custom_objects={'loss': utils.nll_loss(config)})

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
        self.steps_n = 20 # time-steps into the future
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
        state_arr = np.repeat([self.state_t0], self.samples_n, axis=0)
        x = np.zeros([self.samples_n, self.steps_n])
        y = np.zeros([self.samples_n, self.steps_n])
        pc = np.zeros([self.samples_n, self.steps_n])
        pc[:, 0] = state_arr[:, self.scalar_indx['pc_mveh']]

        for t in range(1, self.steps_n):
            mveh_a = self.policy.get_actions(state_arr.copy())
            yveh_a = np.repeat(self.yveh_as[t], self.samples_n, axis=0)
            x[:, t] = x[:, t-1] + state_arr[:, self.scalar_indx['vel_mveh']]*0.1
            y[:, t] = y[:, t-1] + mveh_a[:,1]*0.1
            pc[:, t] = state_arr[:, self.scalar_indx['pc_mveh']]
            state_arr = self.gen.step(state_arr, mveh_a, yveh_a)

        return x, y, pc

    def sceneSetup(self, episode_id):
        m_df, y_df = self.data_obj.get_episode_df(self.val_m_df, self.val_y_df, episode_id)
        v_x_arr, v_y_arr = self.data_obj.get_stateTarget_arr(m_df, y_df)
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
        self.pc_true = m_df['pc'].values[0:self.steps_n]

eval = ModelEvaluation(config)
x, y, pc = eval.trajCompute()
plt.plot(eval.x_true, eval.y_true, color='red')
for i in range(10):
    plt.plot(x[i,:], y[i,:], color='grey')

# %%
for i in range(10):
    plt.plot(pc[i,:], color='grey')
    plt.plot(eval.pc_true, color='red')

# %%
max(eval.pc_true[0:90])
plt.plot(pc[0])
plt.plot(eval.pc_true[0:90])
plt.grid()
# %%

# %%

a = np.array([1,2,3,4,5,6])
np.split(a, 2, axis=0)
import pickle

with open('./datasets/preprocessed/'+'20200921-123920'+'/'+'data_obj', 'rb') as f:
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
