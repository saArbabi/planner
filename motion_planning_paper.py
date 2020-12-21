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
exp_to_evaluate = 'series061exp003'
config = loadConfig(exp_to_evaluate)
# config = loadConfig('series044exp006')
model = MergePolicy(config)
eval_obj = ModelEvaluation(model, config)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2895)

step_sec = 1
end_sec = 4
pred_h = 4
# st_pred = eval_obj.compute_rwse()

# %%
exp_names = 'series060exp008'
dirName = './models/experiments/'+exp_names
with open(dirName+'/'+'rwse_long_lat_vel', 'rb') as f:
    rwse_exp = dill.load(f, ignore=True)

legends = []

for key in rwse_exp.keys():
    if key != 'lat_vel':
        plt.plot(rwse_exp[key])
        legends.append(key)
plt.legend(legends)
a = rwse_exp['vel_fadj']
plt.grid()

# %%
exp_names = 'series060exp009'
dirName = './models/experiments/'+exp_names
with open(dirName+'/'+'rwse_long_lat_vel', 'rb') as f:
    rwse_exp = dill.load(f, ignore=True)

legends = []

for key in rwse_exp.keys():
    if key != 'lat_vel':
        plt.plot(rwse_exp[key])
        legends.append(key)
plt.legend(legends)
b = rwse_exp['vel_fadj']
plt.grid()

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


# %%
#
# exp_names = ['series046exp005', 'series047exp001',
#                                 'series047exp002',
#                                 'series047exp003']
# exp_names = ['series046exp005','series047exp003','series047exp004']
exp_names = ['series046exp005','series047exp003']

    np.ceil(13.3)
# %%
split = 29
for split in range(29, 35):
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





# %%
def choose_traj(self):
    """
    Select best trajectory according to the following metrics:
    - TTC
    - Jerk (mveh and yveh combined)
    - Goal deviation (terminal?)


    """
    trajs = actions[:,:,act_n]
actions.shape
# %%
act_n = 0
mveh_jerk = ((actions[:,1:,act_n] - actions[:,:-1,act_n])/0.1)**2
yveh_jerk = ((actions[:,1:,act_n+2] - actions[:,:-1,act_n+2])/0.1)**2
discount_factor = 0.9
discount = np.power(discount_factor, np.array(range(1,40)))
jerk_cost = np.sum((mveh_jerk+yveh_jerk)*discount, axis=1)

2+2
for i in range(10):
    if i == np.argmin(jerk_cost):
        plt.plot(actions[i,:,act_n], color='green')
    elif i == np.argmax(jerk_cost):
        plt.plot(actions[i,:,act_n], color='red')
    else:
        plt.plot(actions[i,:,act_n], color='grey')

# plt.figure()
# for i in range(10):
#     if i == np.argmin(jerk_cost):
#         plt.plot(actions[i,:,act_n], color='green')
#     elif i == np.argmax(jerk_cost):
#         plt.plot(actions[i,:,act_n], color='red')
#     else:
#         plt.plot(actions[i,:,act_n], color='grey')


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
