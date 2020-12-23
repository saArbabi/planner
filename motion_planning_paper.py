"""
Scripts here were used for creating the figures for paper []
"""
from models.core.train_eval.utils import loadConfig
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from planner import policy
reload(policy)
from planner.policy import TestdataObj, MergePolicy, ModelEvaluation
import dill

# config = loadConfig('series050exp001')
# exp_to_evaluate = 'series054exp002'
# exp_to_evaluate = 'series059exp002'
exp_to_evaluate = 'series063exp001'
config = loadConfig(exp_to_evaluate)
# config = loadConfig('series044exp006')
model = MergePolicy(config)
eval_obj = ModelEvaluation(model, config)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2895)

step_sec = 1
end_sec = 4
pred_h = 4
# st_pred = eval_obj.compute_rwse()
# len(st_pred['lat_vel'])
# %%
exp_names = 'series062exp014'
dirName = './models/experiments/'+exp_names
with open(dirName+'/'+'rwse_long_lat_vel', 'rb') as f:
    rwse_exp = dill.load(f, ignore=True)

legends = []

for key in rwse_exp.keys():
    if key != 'lat_vel':
        plt.plot(rwse_exp[key])
        legends.append(key)


plt.legend(legends)
a = rwse_exp['vel_y']
plt.grid()


plt.figure()
plt.plot(rwse_exp['lat_vel'])
plt.grid()

# %%
exp_names = 'series062exp008'
dirName = './models/experiments/'+exp_names
with open(dirName+'/'+'rwse_long_lat_vel', 'rb') as f:
    rwse_exp = dill.load(f, ignore=True)

legends = []

for key in rwse_exp.keys():
    if key != 'lat_vel':
        plt.plot(rwse_exp[key])
        legends.append(key)
plt.legend(legends)
b = rwse_exp['vel_y']
plt.grid()

plt.figure()
plt.plot(rwse_exp['lat_vel'])
plt.grid()

# %%


# %%
plt.plot(b)
plt.plot(a)

# %%
state_n = 0
for i in range(50):
    plt.plot(st_pred[i,:,state_n], color='grey')
plt.plot(_[:, state_n])
plt.grid()
# %%
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(1289)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2895)
pred_h = 4
st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(1289)
st_i, cond_i, bc_der_i, history_i, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                cond_seq,
                                                st_arr,
                                                targ_arr,
                                                current_step=19,
                                                pred_h=pred_h)


actions = eval_obj.policy.get_actions([st_i, cond_i], bc_der_i, traj_n=10,
                                                                pred_h=pred_h)
actions.shape
# %%
"""How good is my planning and prediction?
"""
act_n = 0
trajs = actions[:,:,act_n]
trajs.shape
avg_traj = np.mean(trajs, axis=0)
st_dev = np.std(trajs, axis=0)
for trj in range(10):
    plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), actions[trj,:,act_n], color='grey')
plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), avg_traj)

plt.plot(np.arange(0, 2, 0.1), history_i[:, act_n], color='orange')
plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), targ_i[:, act_n], color='red')
plt.fill_between(np.arange(1.9, 1.9+pred_h, 0.1), avg_traj+st_dev, avg_traj-st_dev)
plt.grid()
# %%
""" scene evolution plots
"""
for episode in [2895, 1289, 1037, 2870, 2400, 1344, 2872, 2266, 2765, 2215]:
    plt.figure()
    st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(episode)
    plt.plot(targ_arr[:, 1])
    plt.grid()
    plt.title(str(episode))
# %%
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(1037)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2765)
st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2215)

pred_h = 2
for time_step in range(19, 40, 5):

    st_i, cond_i, bc_der_i, history_i, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                    cond_seq,
                                                    st_arr,
                                                    targ_arr,
                                                    current_step=time_step,
                                                    pred_h=pred_h)


    actions = eval_obj.policy.get_actions([st_i, cond_i], bc_der_i, traj_n=50, pred_h=pred_h)
    fig, axs = plt.subplots(1, 5, figsize=(20,3))
    fig.subplots_adjust(wspace=0.05, hspace=0)
    titles = ['Vehicle A',
            'Vehicle A',
            'Vehicle B ',
            'Vehicle C ',
            'Vehicle D ']

    for ax_i in range(5):
        axs[ax_i].set_ylim([-3,3])
        axs[ax_i].set_xlim([-1.9,2.2])
        axs[ax_i].spines['right'].set_visible(False)
        axs[ax_i].spines['top'].set_visible(False)
        axs[ax_i].xaxis.get_major_ticks()[1].label1.set_visible(False)
        axs[ax_i].xaxis.get_major_ticks()[1].label2.set_visible(False)
        if ax_i == 1:
            axs[ax_i].set_ylabel('Lateral action [$m/s$]')
        else:
            axs[ax_i].set_ylabel('Longitudinal action [$m/s^2$]')
        axs[ax_i].set_xlabel('Time [s]')

        if ax_i>0:
            # axs[ax_i].set_yticks([])
            axs[ax_i].set_yticklabels([])


    for act_n in range(5):
        trajs = actions[:,:,act_n]
        avg_traj = np.mean(trajs, axis=0)
        st_dev = np.std(trajs, axis=0)
        axs[act_n].fill_between([-1.9,0],[-3,-3], [3,3], color='lightgrey')

        if time_step == 19:
            axs[act_n].title.set_text(titles[act_n])
        axs[act_n].plot(np.arange(0, pred_h, 0.1), targ_i[:, act_n], color='red')
        axs[act_n].plot(np.arange(-1.9, 0.1, 0.1), history_i[:, act_n], color='black', linewidth=2)
        if act_n < 2:
            axs[act_n].plot(np.arange(0, pred_h, 0.1), avg_traj, color='purple')
            axs[act_n].fill_between(np.arange(0, pred_h, 0.1), avg_traj+st_dev, avg_traj-st_dev, color='lightskyblue')
        for trj in range(50):
            axs[act_n].plot(np.arange(0, pred_h, 0.1), actions[trj,:,act_n], color='grey', linewidth=0.3)


# %%

for rwse_veh in ['long_vel', 'lat_vel']:
    plt.figure()
    plt.title(rwse_veh)
    for exp in exp_names:
        # plt.plot(rwse_exp[exp][rwse_veh])
        plt.plot(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
        plt.scatter(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
    plt.grid()
    plt.legend(exp_names)
# %%
""" rwse plots
"""
fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.subplots_adjust(wspace=0.05, hspace=0)

for ax_i in range(3):
    axs[0,0].set_ylim([3,-3])
    axs[1].set_xlim([0,6.5])
    # axs[ax_i].spines['right'].set_visible(False)
    # axs[ax_i].spines['top'].set_visible(False)
    if ax_i>0:
        # axs[ax_i].set_yticks([])
        axs[ax_i].set_yticklabels([])


# %%

for rwse_veh in ['long_vel', 'lat_vel']:
    plt.figure()
    plt.title(rwse_veh)
    # plt.set_ylim([3,-3])
    plt.xlim([0,3.6])
    plt.xlabel('Horizon (s)')
    if rwse_veh == 'long_vel':
        plt.ylabel('Longitudinal speed (m/s)')
    else:
        plt.ylabel('Lateral speed (m/s)')
    for exp in exp_names:
        # plt.plot(rwse_exp[exp][rwse_veh])
        plt.plot(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
        plt.scatter(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
    plt.grid()
    plt.legend(exp_names)

# %%
"""get rwse
"""
exp_names = ['series057exp003',
            'series057exp004',
            'series057exp005']

for exp_name in exp_names:

    # check if rwse exists, ignore it!
    config = loadConfig(exp_name)
    # config = loadConfig('series044exp006')
    model = MergePolicy(config)
    eval_obj = ModelEvaluation(model, config)
    # eval_obj.compute_rwse()
# %%
"""visualise rwse
"""
"""
Effect of training horizon:
series057exp001: 14 steps-4s
series057exp003: 10 steps-3s
series057exp005: 7 steps-2s
series057exp004: 3 steps-1s
"""

# exp_names = ['series057exp001',
#             'series057exp003',
#             'series057exp005',
#             'series057exp004']
# exp_names = ['series057exp005',
#             'series058exp001',
#             'series059exp001',
#             'series059exp002',
#             'series058exp002']
exp_names = [
            'series060exp007',
            'series059exp001']
rwse_exp = {}
for exp in exp_names:
    dirName = './models/experiments/'+exp
    with open(dirName+'/'+'rwse_long_lat_vel', 'rb') as f:
        rwse_exp[exp] = dill.load(f, ignore=True)


for rwse_veh in ['long_vel', 'lat_vel']:
    plt.figure()
    plt.title(rwse_veh)
    for exp in exp_names:
        # plt.plot(rwse_exp[exp][rwse_veh])
        plt.plot(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
        plt.scatter(np.arange(0, 3.6, 0.5), rwse_exp[exp][rwse_veh][::5])
        # plt.xlim([0, 2.1])
    plt.grid()
    plt.legend(exp_names)

# %%
# %%


# %%
fig_num = 0
pred_h = 4
for episode in [2895, 1289, 1037, 2870, 2400, 1344, 2872, 2266, 2765, 2215]:
    st_seq, cond_seq, _, targ_arr = eval_obj.episodeSetup(episode)
    st_i, cond_i, bc_der_i, _, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                    cond_seq,
                                                    _,
                                                    targ_arr,
                                                    current_step=19,
                                                    pred_h=pred_h)
    actions = eval_obj.policy.get_actions([st_i, cond_i], bc_der_i,
                                                        traj_n=10, pred_h=pred_h)

    for act_n in range(5):
        plt.figure()
        plt.plot(np.arange(0, pred_h, 0.1), targ_i[:, act_n], color='red')

        for trj in range(10):
            plt.plot(np.arange(0, pred_h, 0.1), actions[trj,:,act_n], color='grey')

        plt.grid()
        plt.title(str(fig_num)+'-'+exp_to_evaluate)
        plt.xlabel('Prediction horizon [s]')
        if act_n == 1:
            plt.ylabel('Lateral speed [m/s]')
        else:
            plt.ylabel('Acceleration [$m/s^2$]')
        fig_num += 1

# %%
config = loadConfig('series025exp001')
test_data = TestdataObj(config)
model = MergePolicy(config)
eval_obj = ModelEvaluation(model, config)
for episode_id in [2895, 1289]:
    vis(episode_id)
# %%
# eval_obj.compute_rwse()
 
