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

config = loadConfig('series046exp005')
# config = loadConfig('series044exp006')
model = MergePolicy(config)
eval_obj = ModelEvaluation(model, config)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2895)

step_sec = 1
end_sec = 4
pred_h = 4

# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(1289)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2895)
st_seq, cond_seq, targ_arr = eval_obj.episodeSetup(1289)
st_i, cond_i, bc_der_i, targ_i = eval_obj.sceneSetup(st_seq,
                                                cond_seq,
                                                targ_arr,
                                                current_step=19,
                                                pred_h=pred_h)
targ_i.shape
actions = eval_obj.policy.get_actions([st_i, cond_i], bc_der_i, traj_n=10, pred_h=pred_h)
# %%
"""How good is my planning and prediction?
"""
act_n = 4
trajs = actions[:,:,act_n]
trajs.shape
avg_traj = np.mean(trajs, axis=0)
st_dev = np.std(trajs, axis=0)
for trj in range(5):
    plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), actions[trj,:,act_n], color='grey')
plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), avg_traj)
plt.plot(np.arange(0, 1.9+pred_h, 0.1), targ_i[:, act_n], color='red')
plt.fill_between(np.arange(1.9, 1.9+pred_h, 0.1), avg_traj+st_dev, avg_traj-st_dev)
# %%

for act_n in range(5):
    plt.figure()
    plt.plot(np.arange(0, 1.9+pred_h, 0.1), targ_i[:, act_n], color='red')

    for trj in range(10):
        plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), actions[trj,:,act_n], color='grey')

    plt.grid()
    plt.title('time_step: '+ str(time_step))
    plt.xlabel('Prediction horizon [s]')
    if act_n == 1:
        plt.ylabel('Lateral speed [m/s]')
    else:
        plt.ylabel('Acceleration [$m/s^2$]')
# %%
for time_step in range(19, 59, 5):
# for time_step in range(19,20):
    st_i, cond_i, bc_der_i, targ_i = eval_obj.sceneSetup(st_seq,
                                                    cond_seq,
                                                    targ_arr,
                                                    current_step=time_step,
                                                    pred_h=pred_h)
    actions = eval_obj.policy.get_actions([st_i, cond_i], bc_der_i, traj_n=10, pred_h=pred_h)

    for act_n in range(5):
        plt.figure()
        plt.plot(np.arange(0, 1.9+pred_h, 0.1), targ_i[:, act_n], color='red')

        for trj in range(10):
            plt.plot(np.arange(1.9, 1.9+pred_h, 0.1), actions[trj,:,act_n], color='grey')

        plt.grid()
        plt.title('time_step: '+ str(time_step))
        plt.xlabel('Prediction horizon [s]')
        if act_n == 1:
            plt.ylabel('Lateral speed [m/s]')
        else:
            plt.ylabel('Acceleration [$m/s^2$]')
# %%


# %%
fig_num = 0
for episode in [2895, 1289, 1037, 2870, 2400, 1344, 2872, 2266, 2765, 2215]:
    st_seq, cond_seq, targ_arr = eval_obj.episodeSetup(episode)
    bc_der = (targ_arr[19, :]-targ_arr[18, :])*10

    actions = eval_obj.policy.get_actions([st_seq[0,:,:],
                                    [cond_seq[n][0,:,:] for n in range(5)]], bc_der, 10, pred_h)

    for act_n in range(5):
        plt.figure()
        plt.plot(np.arange(0, pred_h, 0.1), targ_arr[19:19+40, act_n], color='red')

        for trj in range(10):
            plt.plot(np.arange(0, pred_h, 0.1), actions[trj,:,act_n], color='grey')

        plt.grid()
        plt.title(str(fig_num))
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
