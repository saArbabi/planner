import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
from models.core.preprocessing import data_prep
DataObj = data_prep.DataObj

from models.core.tf_models import utils
import matplotlib.pyplot as plt
from importlib import reload

from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
# %%

# %%
reload(data_prep)
DataObj = data_prep.DataObj

class GenModel():
    """TODO:
    - reward (A*state + B*Belief)
    - can be attr of the mcts planner
    """
    def __init__(self):
        self.steps_n = 10 # time-steps into the future
        self.step_size = 0.1

    def trajCompute(self, state_t0, config):
        """
        :Param:
            state0 - current vehicle state
        :Return: [[x_cor, y_cor], ...]
        """
        state_arr = np.repeat([state_t0], self.samples_n, axis=0)
        for t in range(self.steps_n):
            state_arr = self.step(state_arr, t)
            self.x.append(self.x[-1] + state_arr[:, self.mvehindx['vel']]*0.1)
            self.y.append(self.y[-1] + act_lat*0.1)

        return self.x, self.y

    def updateArr(self, state_arr, mveh_a, yveh_a):
        """Updates the state values the result from interacting vehicles taking action
        """
        indx = self.retain_indx['vel_mveh']
            state_arr[:,indx] += mveh_a[:,0]*self.step_size

        indx = self.retain_indx['vel_yveh']
            state_arr[:,indx] += mveh_a[:,0]*self.step_size

        for state_key in self.mveh_s['boolean']:
            indx = self.bool_indx[state_key+'_mveh']

        for state_key in self.yveh_s['boolean']:
            indx = self.bool_indx[state_key+'_yveh']

        for state_key in self.mveh_s['scalar']:
            indx = self.scalar_indx[state_key+'_mveh']
            if state_key == 'vel':
                state_arr[:,indx] += mveh_a[:,0]*self.step_size
            if state_key == 'pc':
                state_arr[:, indx] += act_lat*self.step_size
                lc_left = state_arr[:, indx]  > 1.85
                state_arr[lc_left, indx] = -1.85
                lc_right = state_arr[:, indx]  < -1.85
                state_arr[lc_right, indx] = 1.85
            # if state_key == 'dx':

        for state_key in self.yveh_s['scalar']:
            indx = self.scalar_indx[state_key+'_yveh']


    def step(self, state_arr, policy, driver_model):

        for time_step in range(self.steps_n):
            mveh_a = policy.get_actions(state_arr)
            yveh_a = driver_model.get_actions(time_step)

        return state_arr

class YieldModel():
    def __init__(self, config):
        # self.loadModel(config)
        self.data_obj = DataObj(config)
        self.mvehindx = self.data_obj.mvehindx
        self.yvehindx = self.data_obj.yvehindx


class MergePolicy():
    def __init__(self, config):
        self.loadModel(config)
        self.data_obj = DataObj(config)
        self.mvehindx = self.data_obj.mvehindx
        traj_n = 10


        # self.ego = ego

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
        action_samples = utils.get_pdf_samples(1, parameter_vector, config['model_type'])
        actions = self.data_obj.applyInvScaler(action_samples)

        return  actions.reshape(actions.shape[1], actions.shape[-1])

# %%
config = loadConfig('exp003')
controller = MergePolicy(config)

m_df, y_df = controller.data_obj.get_episode_df(811)
x_arr, y_arr = controller.data_obj.get_stateTarget_arr(m_df, y_df)

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

# %%
obs in
utils.get_pdf_samples
config = loadConfig('exp003')
model = loadModel(config)


# %%
# model.predict(x_arr[0])






    # stateii = statei.copy()
    stateii = statei.copy()
    stateii[['a_long_c','v_lat_c']] = a_long_ii, v_lat_ii

    v_long_i = statei['v_long']
    v_long_ii = v_long_i + a_long_ii*step


    ff_long_ii = statei['ff_long'] + statei['ff_v']*step
    bb_long_ii = statei['bb_long'] + statei['bb_v']*step

    if ff_long_ii < 0:
        ff_long_ii = 0
    elif ff_long_ii > 70:
        ff_long_ii = 70

    if bb_long_ii < 0:
        bb_long_ii = 0
    elif bb_long_ii > 70:
        bb_long_ii = 70

    pc_ii = statei['pc'] + v_lat_ii*step
    ff_v_ii = frm_ps_val_set['ff_v_long'] - v_long_ii
    bb_v_ii = v_long_ii - frm_ps_val_set['bb_v_long']
    stateii['ff_v'] = ff_v_ii
    stateii['bb_v'] = bb_v_ii
    stateii['ff_bb_v'] = frm_ps_val_set['ff_bb_v']

    stateii['lc_bool'] = frm_ps_val_set['lc_bool']
    stateii['frm'] = frm_ps_val_set['frm']
    stateii[['ff_v_long','bb_v_long']] = frm_ps_val_set[['ff_v_long','bb_v_long']]
    stateii[['ff_a','bb_a']] = frm_ps_val_set[['ff_a','bb_a']]

    stateii[['a_long_p','v_lat_p']] = a_long_ii, v_lat_ii

    stateii['v_long'] = v_long_ii
    stateii['ff_long'] = ff_long_ii
    stateii['bb_long'] = bb_long_ii
    stateii['pc'] = pc_ii

    return stateii



state_arr[0,:] = 0
trajCollection = []
for step in range(trajSteps_n):
