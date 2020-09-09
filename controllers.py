import numpy as np
import pandas as pd
from models.core.train_eval.utils import loadConfig
from models.core.preprocessing.data_prep import DataObj
from models.core.tf_models import utils
import matplotlib.pyplot as plt

# %%
from tensorflow import keras


# %%
class MergeController():
    def __init__(self, config):
        self.loadModel(config)
        self.loadScalers(config)
        self.setState_indx(config)
        # self.ego = ego

    def loadModel(self, config):
        dirName = './models/experiments/'+config['exp_id'] +'/trained_model'
        self.model = keras.models.load_model(dirName,
                                    custom_objects={'loss': utils.nll_loss(config)})

    def loadScalers(self, config):
        my_data = DataObj(config)
        self.state_scaler = my_data.state_scaler
        self.target_scaler = my_data.target_scaler

    def trajCompute(self, config):
        pass

    def setState_indx(self, config):
        veh_states = config['data_config']['veh_states']
        i = 0
        self.mvehindx = {}
        self.yvehindx = {}

        for state_key in veh_states['mveh']:
            self.mvehindx[state_key] = i
            i += 1

        for state_key in veh_states['yveh']:
            self.yvehindx[state_key] = i
            i += 1

    def get_actions(self, state_arr, samples_n):
        """
        :Param: Unscaled state vectors
        :Return: Arrays for the computed control actions
        """
        my_data.state_scaler.transform(state_arr)
        parameter_vector = self.model.predict(state_arr.reshape(1,-1))
        action_samples = utils.get_pdf_samples(samples_n, parameter_vector, config)
        actions = my_data.target_scaler.inverse_transform(action_samples)
        act_long = np.array([a[0] for a in actions.reshape(-1, 2)])
        act_lat = np.array([a[1] for a in actions.reshape(-1, 2)])

        return act_long, act_lat

    def 


config = loadConfig('exp003')
model = loadModel(config)
x_arr, y_arr = my_data.episode_prep(811)
my_data.target_scaler.inverse_transform(y_arr[0])
# %%
from importlib import reload
reload(utils)
config['data_config']['veh_states'].values()
list(config['data_config']['veh_states'].values())




state_arr

state_arr[:, s_indx['vel']] += act_long*0.1


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
