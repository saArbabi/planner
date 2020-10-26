import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
import json
import matplotlib.pyplot as plt
from importlib import reload
import os
from models.core.tf_models.cae_model import CAE
import pickle
import tensorflow as tf
import dill
from collections import deque

from models.core.tf_models import utils
reload(utils)

# %%

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
                            shape:([steps, traj_n, all-actions])

        :Return: Predicted future states
        """
        state_predictions = [st_arr]
        state_i = st_arr.copy()
        for step in range(steps_n-1):
            st_arr_ii = self.step(state_i, acts_arr[step, :, :])
            state_predictions.append(st_arr_ii)
            state_i = st_arr_ii

        return np.array(state_predictions)

class MergePolicy():
    def __init__(self, data_obj, config):
        self.data_obj = data_obj
        self.loadModel(config)

        # TODO:
        # objective function/ evaluate function/ set execution time, which will
        # execute best ranked traj for a period of time.

    def loadModel(self, config):
        checkpoint_dir = './models/experiments/'+config['exp_id'] +'/model_dir'
        self.model = CAE(config, model_use='inference')
        Checkpoint = tf.train.Checkpoint(net=self.model)
        # Checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        Checkpoint.restore(checkpoint_dir+'/ckpt-10')

        self.enc_model = self.model.enc_model
        self.dec_model = self.model.dec_model

    def get_actions(self, dec_inputs, traj_n, steps_n):
        """
        :Return: unscaled action array for all cars
        """
        self.dec_model.steps_n = steps_n
        self.dec_model.traj_n = traj_n

        gmm_m, gmm_y, gmm_f, gmm_fadj = self.dec_model(dec_inputs)
        total_acts_count = traj_n*steps_n
        veh_acts_count = 5 # 2 for merging, 1 for each of the other cars

        unscaled_acts = np.zeros([total_acts_count, veh_acts_count])
        veh_acts = gmm_m.sample().numpy()
        veh_acts.shape = (total_acts_count, 2)
        i = 2
        unscaled_acts[:, 0:i] = veh_acts
        for gmm in [gmm_y, gmm_f, gmm_fadj]:
            veh_acts = gmm.sample().numpy()
            veh_acts.shape = (total_acts_count)
            unscaled_acts[:, i] = veh_acts
            i += 1

        scaled_acts = self.data_obj.apply_InvScaler(unscaled_acts)
        scaled_acts.shape = (traj_n, steps_n, veh_acts_count)

        return np.stack(scaled_acts, axis=1) # shape: [steps, traj_n, all-actions]

class ModelEvaluation():
    dirName = './datasets/preprocessed/'

    def __init__(self, config):
        self.data_config = config['data_config']
        self.setup() # load data_obj and validation data
        self.policy = MergePolicy(self.data_obj, config)
        self.gen_model = GenModel()
        self.sceneSetup(episode_id=2895)
        # self.sceneSetup(episode_id=1289)

    def setup(self):
        config_names = os.listdir(self.dirName+'config_files')
        for config_name in config_names:
            with open(self.dirName+'config_files/'+config_name, 'r') as f:
                config = json.load(f)

            if config == self.data_config:
                with open(self.dirName+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = dill.load(f, ignore=True)

                with open(self.dirName+config_name[:-5]+'/'+'states_test', 'rb') as f:
                    self.states_test = pickle.load(f)

                with open(self.dirName+config_name[:-5]+'/'+'targets_test', 'rb') as f:
                    self.targets_test = pickle.load(f)

                with open(self.dirName+config_name[:-5]+'/'+'conditions_test', 'rb') as f:
                    self.conditions_test = pickle.load(f)

    def obsSequence(self, state_arr, target_arr, condition_arr):
        state_seq = []
        target_seq = []
        condition_seq = []

        step_size = 1
        pred_horizon = 20
        obsSequence_n = 20
        i_reset = 0
        i = 0
        for chunks in range(step_size):
            prev_states = deque(maxlen=obsSequence_n)
            while i < (len(state_arr)-2):
                # 2 is minimum prediction horizon
                prev_states.append(state_arr[i])
                if len(prev_states) == obsSequence_n:
                    state_seq.append(np.array(prev_states))
                    target_seq.append(target_arr[i:i+pred_horizon].tolist())
                    condition_seq.append(condition_arr[i:i+pred_horizon].tolist())

                i += step_size
            i_reset += 1
            i = i_reset

        return state_seq, target_seq, condition_seq

    def sceneSetup(self, episode_id):
        self.true_st_arr = self.states_test[self.states_test[:, 0] == episode_id][:, 1:]
        self.true_target_arr = self.targets_test[self.targets_test[:, 0] == episode_id][:, 1:]
        condition_arr = self.conditions_test[self.conditions_test[:, 0] == episode_id][:, 1:]

        st_arr = self.data_obj.applystateScaler(self.true_st_arr.copy())
        condition_arr = self.data_obj.applyconditionScaler(condition_arr)
        self.st_seq, _, self.condition_seq = self.obsSequence(
                                            st_arr, self.true_target_arr, condition_arr)

    def trajCompute(self, traj_n, steps_n):
        obs_seq = self.st_seq[0]
        seq_shape = obs_seq.shape
        obs_seq.shape = (1, seq_shape[0], seq_shape[1])
        obs_seq = np.repeat(obs_seq, traj_n, axis=0)
        conditions = np.array(self.condition_seq[0])
        seq_shape = conditions.shape
        conditions.shape = (1, seq_shape[0], seq_shape[1])
        conditions = np.repeat(conditions, traj_n, axis=0)

        # compute actions
        enc_state = self.policy.enc_model(obs_seq)
        actions = self.policy.get_actions([conditions, enc_state], traj_n, steps_n)
        # simulate state forward
        state_i = np.repeat([self.true_st_arr[19]], traj_n, axis=0)
        self.gen_model.max_pc = max(self.true_st_arr[:, self.gen_model.indx_m['pc']])
        self.gen_model.min_pc = min(self.true_st_arr[:, self.gen_model.indx_m['pc']])
        state_predictions = self.gen_model.forwardSim(state_i, actions, steps_n)
        # get xy poses for mveh

        return state_predictions

# config = loadConfig('series007exp001')
config = loadConfig('series003exp002')
eval_obj = ModelEvaluation(config)

traj_n = 10
steps_n = 20
pred_st = eval_obj.trajCompute(traj_n, steps_n)
pred_st.shape

# %%


# %%
def plot_state(vis_state):
    plt.plot(range(steps_n), eval_obj.true_st_arr[19:19+steps_n, vis_state], color='red')

    for n in range(traj_n):
        plt.plot(pred_st[:, n, vis_state], color='grey')
    plt.grid()
    plt.xlabel('steps')
    plt.ylabel('longitudinal speed [m/s]')
    plt.title('Merge vehicle trajectory')

vis_state_m = eval_obj.gen_model.indx_m['vel']
plot_state(vis_state_m)
plot_state(eval_obj.gen_model.indx_y['vel'])
plot_state(eval_obj.gen_model.indx_f['vel'])
plot_state(eval_obj.gen_model.indx_fadj['vel'])




# %%
pred_st.shape
plt.plot(range(steps_n), eval_obj.true_st_arr[19:19+steps_n, eval_obj.gen_model.indx_m['pc']], color='red')

for n in range(traj_n):
    plt.plot(pred_st[:, n, eval_obj.gen_model.indx_m['pc']], color='grey')

plt.grid()
plt.xlabel('steps')
plt.ylabel('pc [m]')
plt.title('Merge vehicle trajectory')

# %%


# %%
# %%
"""State plots
"""
for item in eval.scalar_indx:
    fig = plt.figure()
    plt.plot(eval.st_arr_true[:, eval.scalar_indx[item]], color='red')
    for s in range(eval.samples_n):
        plt.plot(st_arr[s, :, eval.scalar_indx[item]], color='grey')
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
