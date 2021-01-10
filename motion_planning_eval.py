from models.core.train_eval.utils import loadConfig
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from planner import policy
reload(policy)
from planner.policy import TestdataObj, MergePolicy, ModelEvaluation
import dill

exp_to_evaluate = 'series081exp001'
config = loadConfig(exp_to_evaluate)
traffic_density = ''
# traffic_density = 'high_densit_'
traffic_density = 'medium_density_'
# traffic_density = 'low_density_'
test_data = TestdataObj(traffic_density, config)

model = MergePolicy(test_data, config)
eval_obj = ModelEvaluation(model, test_data, config)
eval_obj.compute_rwse(traffic_density)

# %%
"""To visualise rwse against prediction horizon
"""
exps = [
        # 'series077exp001',
        # 'series078exp001',
        'series079exp002',
        'series081exp001',

        ]
densities = ['high_density_']
# densities = ['medium_density_']
# densities = ['low_density_']

discounted_exp_results = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for key in ['vel_m','lat_vel','vel_y','vel_f','vel_fadj']:
    legends = []
    plt.figure()
    for exp_name in exp_names:
        plt.plot(rwses[exp_name][key])
        legends.append(key+'_'+exp_name)
    plt.legend(legends)
    plt.grid()

# %%
fig_num = 0
pred_h = 4
# for episode in [2895, 1289]:
# for episode in [2895, 1289, 1037]:
for episode in [2895, 1289, 1037, 2870, 2400, 1344, 2872, 2266, 2765, 2215]:
    st_seq, cond_seq, _, targ_arr = eval_obj.episodeSetup(episode)
    st_i, cond_i, bc_der_i, _, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                    cond_seq,
                                                    _,
                                                    targ_arr,
                                                    current_step=19,
                                                    pred_h=pred_h)
    actions, _, _ = eval_obj.policy.get_actions([st_i.copy(), cond_i.copy()], bc_der_i,
                                                        traj_n=10, pred_h=pred_h)

    for act_n in range(5):
        plt.figure()
        plt.plot(np.arange(0, pred_h+0.1, 0.1), targ_i[:, act_n], color='red')

        for trj in range(10):
            plt.plot(np.arange(0, pred_h+0.1, 0.1), actions[trj,:,act_n], color='grey')

        # plt.grid()
        plt.title(str(fig_num)+'-'+exp_to_evaluate)
        plt.xlabel('Prediction horizon [s]')
        if act_n == 1:
            plt.ylabel('Lateral speed [m/s]')
        else:
            plt.ylabel('Acceleration [$m/s^2$]')
        fig_num += 1
