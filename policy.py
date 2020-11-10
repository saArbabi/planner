import numpy as np
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
    def __init__(self, config):
        self.loadModel(config)
        self.test_data = TestdataObj(config)

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

    def get_actions(self, seq, traj_n, steps_n):
        """
        :Param: [state_seq, cond_seq], traj_n, steps_n
        :Return: unscaled action array for all cars
        """
        st_seq, cond_seq = seq
        # reshape to fit model
        st_seq.shape = (1, st_seq.shape[0], st_seq.shape[1])
        cond_seq.shape = (1, 1, 5)
        st_seq = np.repeat(st_seq, traj_n, axis=0)
        cond_seq = np.repeat(cond_seq, traj_n, axis=0)

        # get enc_h state
        enc_state = self.enc_model(st_seq)

        self.dec_model.steps_n = steps_n
        self.dec_model.traj_n = traj_n

        gmm_m, gmm_y, gmm_f, gmm_fadj = self.dec_model([cond_seq, enc_state])
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

        scaled_acts = test_data.data_obj.apply_InvScaler(unscaled_acts)
        scaled_acts.shape = (traj_n, steps_n, veh_acts_count)

        return scaled_acts

class TestdataObj():
    dirName = './datasets/preprocessed/'
    def __init__(self, config):
        self.setup(config['data_config']) # load test_data and validation data

    def setup(self, data_config):
        self.test_episodes = np.loadtxt('./datasets/test_episodes.csv', delimiter=',')
        config_names = os.listdir(self.dirName+'config_files')
        for config_name in config_names:
            with open(self.dirName+'config_files/'+config_name, 'r') as f:
                config = json.load(f)

            if config == data_config:
                with open(self.dirName+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = dill.load(f, ignore=True)

                with open(self.dirName+config_name[:-5]+'/'+'states_test', 'rb') as f:
                    self.states_set = pickle.load(f)

                with open(self.dirName+config_name[:-5]+'/'+'targets_test', 'rb') as f:
                    self.targets_set = pickle.load(f)

                with open(self.dirName+config_name[:-5]+'/'+'conditions_test', 'rb') as f:
                    self.conditions_set = pickle.load(f)


class ModelEvaluation():
    def __init__(self, model, config):
        self.policy = model
        self.test_data = TestdataObj(config)
        self.gen_model = GenModel()
        self.episode_n = 50
        self.steps_n = 40
        self.traj_n = 20
        self.dirName = './models/experiments/'+config['exp_id']

    def obsSequence(self, st_arr, cond_arr, test_data):
        st_arr = test_data.data_obj.applystateScaler(st_arr)
        cond_arr = test_data.data_obj.applyconditionScaler(cond_arr)

        st_seq = {}
        cond_seq = {}

        i_reset = 0
        i = 0
        for chunks in range(test_data.data_obj.step_size):
            prev_states = deque(maxlen=test_data.data_obj.obsSequence_n)
            while i < (len(st_arr)-self.steps_n):
                prev_states.append(st_arr[i])
                if len(prev_states) == test_data.data_obj.obsSequence_n:
                    # i is used as a key for retrieving corresponding true targets
                    st_seq[i] = np.array(prev_states)
                    cond_seq[i] = cond_arr[i:i+1]

                i += test_data.data_obj.step_size
            i_reset += 1
            i = i_reset

        return st_seq, cond_seq

    def sceneSetup(self, episode_id, test_data):
        """:Return: All info needed for evaluating model on a given scene
        """
        st_arr = test_data.states_set[test_data.states_set[:, 0] == episode_id][:, 1:]
        targ_arr = test_data.targets_set[test_data.targets_set[:, 0] == episode_id][:, 1:]
        cond_arr = test_data.conditions_set[test_data.conditions_set[:, 0] == episode_id][:, 1:]
        st_seq, cond_seq = self.obsSequence(st_arr.copy(), cond_arr.copy(), test_data)

        return st_seq, cond_seq, st_arr, targ_arr

    def root_weightet_sqr(self, true_traj, pred_trajs):
        err = np.sqrt(np.mean(np.square(true_traj-pred_trajs), axis=0))
        return err

    def trajCompute(self, episode_id):
        st_seq, cond_seq, st_arr, targ_arr = self.sceneSetup(episode_id, self.test_data)
        actions = self.policy.get_actions([st_seq[29].copy(), cond_seq[29].copy()],
                                                    self.traj_n, self.steps_n)

        # simulate state forward
        state_i = np.repeat([st_arr[29, :]], self.traj_n, axis=0)
        self.gen_model.max_pc = max(st_arr[:, self.gen_model.indx_m['pc']])
        self.gen_model.min_pc = min(st_arr[:, self.gen_model.indx_m['pc']])
        state_predictions = self.gen_model.forwardSim(state_i.copy(), actions, self.steps_n)
        state_true = st_arr[0:29+self.steps_n]
        return state_true, state_predictions

    def compute_rwse(self):
        """
        dumps dict into exp folder containing RWSE for all vehicle actions across time.
        """
        rwse_dict = {'m_long':0, 'm_lat':1, 'y_long':2, 'f_long':3, 'fadj_long':4}
        pred_arrs = [np.zeros([self.episode_n*self.traj_n*6, self.steps_n]) for i in range(5)]
        truth_arrs = [np.zeros([self.episode_n*self.traj_n*6, self.steps_n]) for i in range(5)]
        _row = 0

        for episode_id in self.test_data.test_episodes[:self.episode_n]:
            st_seq, cond_seq, _, targ_arr = self.sceneSetup(episode_id, self.test_data)
            if len(st_seq) > 6:
                splits_n = 6
            else:
                splits_n = len(st_seq)

            traj_splits = np.random.choice(list(st_seq.keys()), splits_n, replace=False)

            for split in traj_splits:

                st_seq_i = st_seq[split]
                cond_seq_i = cond_seq[split]
                true_arr_i = targ_arr[split:split+self.steps_n, :]
                true_arr_i.shape = (1, self.steps_n, 5)
                actions = self.policy.get_actions([st_seq_i.copy(), cond_seq_i.copy()],
                                                        self.traj_n, self.steps_n)

                for vehact_indx in range(5):
                    # veh_mlong, veh_mlat, veh_y, veh_f, veh_fadj
                    truth_arrs[vehact_indx][_row:_row+self.traj_n, :] = true_arr_i[:,:,vehact_indx]
                    pred_arrs[vehact_indx][_row:_row+self.traj_n, :] = actions[:,:,vehact_indx]
                _row += self.traj_n

        for key in rwse_dict.keys():
            rwse_dict[key] = self.root_weightet_sqr(truth_arrs[rwse_dict[key]], \
                                                        pred_arrs[rwse_dict[key]])


        with open(self.dirName+'/rwse', "wb") as f:
            pickle.dump(rwse_dict, f)

config = loadConfig('series022exp003')
test_data = TestdataObj(config)
model = MergePolicy(config)
eval_obj = ModelEvaluation(model, config)

# %%
# eval_obj.compute_rwse()


# %%
exp_list = ['series022exp001', 'series022exp002', 'series022exp003']
rwse_exp = {}
for exp in exp_list:
    dirName = './models/experiments/'+exp
    with open(dirName+'/'+'rwse', 'rb') as f:
        rwse_exp[exp] = dill.load(f, ignore=True)

# %%

for exp in exp_list:
    plt.plot(rwse_exp[exp]['m_long'])
plt.legend(exp_list)
# %%
split = 19
for split in range(19, 35):
    plt.figure()
    actions = eval_obj.policy.get_actions([st_seq[split].copy(), cond_seq[split].copy()], 5, 30)
    plt.plot(targ_arr[split:split+30 ,0])
    for traj in actions[:,:,0]:
        plt.plot(traj)


# %%
pred_arrs, truth_arrs = eval_obj.compute_rwse()

pred_arrs[0][0:2].shape
plt.plot(pred_arrs[0][3,:])
plt.plot(truth_arrs[0][1,:])

# %%
_row = 0

for _traj in range(18):
    plt.figure()
    plt.title(str(_traj+1))

    for sample_traj in range(_traj, _traj+2):
        plt.plot(pred_arrs[0][_row,:])
        _row += 1
    plt.grid()
    plt.plot(truth_arrs[0][_traj,:])

# %%
plt.plot(err_arrs['m_long'])

plt.plot(err_arrs['y_long'])
plt.plot(err_arrs['f_long'])

# %%


# a = np.array([[2,4,6], [3,5,7]])
# b = np.array([1,1,1])
#
# a-b
# config = loadConfig('series020exp001')
# eval_obj = ModelEvaluation(config)
# state_true, state_predictions = eval_obj.trajCompute(2895)

# %%
err_arrs = eval_obj.compute_rwse()
for i in range(5):
    plt.plot(err_arrs[i])
plt.legend([1,2,3,4,5])

# %%
np.linspace()

discount_factor = 0.9
np.power(discount_factor, np.array(range(1,20)))



# %%
def choose_traj(self):
    """
    Select best trajectory according to the following metrics:
    - TTC
    - Jerk (mveh and yveh combined)
    - Goal deviation (terminal?)


    """
# %%


# %%
# config = loadConfig('series007exp001')

# episode_id = 2895
def vis(episode_id):
    state_true, state_predictions = eval_obj.trajCompute(episode_id)

    mid_point = 29
    end_point = 29+eval_obj.steps_n
    title_info = 'episode_id: ' + str(episode_id) + ' max_err: ' + str(config['model_config']['allowed_error'])
    fig, axs = plt.subplots(1, 3, figsize=(15,3))

    vis_state = eval_obj.gen_model.indx_m['vel']
    axs[0].plot(range(end_point), state_true[0:end_point, vis_state], color='red')

    for n in range(eval_obj.traj_n):
        axs[0].plot(range(mid_point, end_point), state_predictions[n, :, vis_state], color='grey')
    axs[0].grid()
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('longitudinal speed [m/s]')
    axs[0].set_title('mveh' + title_info)

    vis_state = eval_obj.gen_model.indx_y['vel']
    axs[1].plot(range(end_point), state_true[0:end_point, vis_state], color='red')

    for n in range(eval_obj.traj_n):
        axs[1].plot(range(mid_point, end_point), state_predictions[n, :, vis_state], color='grey')
    axs[1].grid()
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('longitudinal speed [m/s]')
    axs[1].set_title('yveh ' + title_info)

    vis_state_m = eval_obj.gen_model.indx_m['pc']
    axs[2].plot(range(end_point), state_true[0:end_point, eval_obj.gen_model.indx_m['pc']], color='red')
    state_true[-1]
    for n in range(eval_obj.traj_n):
        axs[2].plot(range(mid_point, end_point), state_predictions[n, :, eval_obj.gen_model.indx_m['pc']] , color='grey')

    axs[2].grid()
    axs[2].set_xlabel('steps')
    axs[2].set_ylabel('pc [m]')
    axs[2].set_title('mveh ' + title_info)

    fig.tight_layout()

for episode_id in [2895, 1289]:
    vis(episode_id)


# %%
ckpt
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
