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

exp_to_evaluate = 'series081exp003'
config = loadConfig(exp_to_evaluate)
traffic_density = ''
# traffic_density = 'high_densit_'
# traffic_density = 'medium_density_'
# traffic_density = 'low_density_'
test_data = TestdataObj(traffic_density, config)

model = MergePolicy(test_data, config)
eval_obj = ModelEvaluation(model, test_data, config)
eval_obj.compute_rwse(traffic_density)

st_seq, cond_seq, _, targ_arr = eval_obj.episodeSetup(episode)
st_i, cond_i, bc_der_i, _, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                cond_seq,
                                                _,
                                                targ_arr,
                                                current_step=19,
                                                pred_h=pred_h)
actions, prob_mlon, prob_mlat = eval_obj.policy.get_actions([st_i.copy(), cond_i.copy()], bc_der_i, traj_n=50, pred_h=pred_h)
# prob_mlon.shape

# %%
"""Compare rwse for different architectures and traffic densities
"""
discount_factor = 0.9
gamma = np.power(discount_factor, np.array(range(0,20)))

exps = [
        # 'series077exp001', # baseline
        # 'series078exp001', # only target car in conditional = to show interactions mater
        'series079exp002', # no teacher helping - to show it maters
        'series081exp001',
        ]
densities = ['low_density_','medium_density_', 'high_density_']

rwses = {}
for exp_i in range(len(exps)):
    for density_i in range(len(densities)):
        dirName = './models/experiments/'+exps[exp_i]+'/'+densities[density_i]+'rwse'
        with open(dirName, 'rb') as f:
            rwses[exps[exp_i]+densities[density_i]] = dill.load(f, ignore=True)

# %%
exps = [
        # 'series077exp001',
        # 'series078exp001',
        'series079exp002',
        'series081exp001',

        ]
densities = ['high_density_']
densities = ['medium_density_']
# densities = ['low_density_']

discounted_exp_results = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for exp_name in exp_names:
    discounted_exp_results[exp_name] = []

    for key in ['vel_m','lat_vel','vel_y','vel_fadj', 'vel_f']:
        discounted_exp_results[exp_name].append(np.sum(rwses[exp_name][key]*gamma))
# %%
"""To visualise rwse against prediction horizon
"""
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
"""Bar chart visualistation
"""
labels = ['$\dot x_{0}$', '$\dot y_{0}$', '$\dot x_{1}$', '$\dot x_{2}$', '$\dot x_{3}$']
exp1 = discounted_exp_results[exp_names[0]]
exp2 = discounted_exp_results[exp_names[1]]
exp3 = discounted_exp_results[exp_names[2]]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, exp1, width,
                                color='lightgrey', edgecolor='black', hatch='//')
rects2 = ax.bar(x + width, exp2, width,
                                color='grey', edgecolor='black')
rects2 = ax.bar(x - width, exp3, width,
                                color='grey', edgecolor='black', hatch='//')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time discounted RWSE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', alpha=0.3)


fig.tight_layout()

plt.show()
fig.savefig("low_density_performance.png", dpi=200)

# %%

# %%
"""Visualise distributions
"""
pred_h = 2
st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2215)

st_i, cond_i, bc_der_i, history_i, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                cond_seq,
                                                st_arr,
                                                targ_arr,
                                                current_step=9,
                                                pred_h=pred_h)


actions = eval_obj.policy.get_actions([st_i.copy(), cond_i.copy()], bc_der_i, traj_n=50, pred_h=pred_h)
actions.shape
##############
act_n = 0
# ground_truth = targ_arr[22:22+40, act_n][::3]
ground_truth = targ_arr[20:20+40, act_n]
action_range = np.linspace(-5, 5, num=200)
fig_num = 0
for time_step in range(40):
    plt.figure()
    likelihood_1 = gmm_mlons[0,time_step].prob(action_range)
    likelihood_2 = gmm_mlons[1,time_step].prob(action_range)
    # likelihood_3 = gmm_mlons[2,time_step].prob(action_range)
    plt.plot(action_range, likelihood_1)
    plt.scatter(actions[0,time_step,act_n], gmm_mlons[0,time_step].prob(
                                    actions[0,time_step,act_n]), marker='x')

    plt.plot(action_range, likelihood_2)
    plt.scatter(actions[1,time_step,act_n], gmm_mlons[1,time_step].prob(
                                    actions[1,time_step,act_n]), marker='x')

    fig_num += 1
    plt.title(str(fig_num))

    plt.grid()
# %%
""" rwse against training horizon

series077exp008 - 1 steps
series077exp005 - 3 steps
series077exp001 - 7 steps
series077exp004 - 10 steps
series077exp003 - 13 steps
"""

exps = [
        'series077exp008',
        'series077exp005',
        'series077exp001',
        'series077exp004',
        'series077exp003',
        ]
densities = ['medium_density_']
# densities = ['high_density_']

rwses = {}
for exp_i in range(len(exps)):
    for density_i in range(len(densities)):
        dirName = './models/experiments/'+exps[exp_i]+'/'+densities[density_i]+'rwse'
        with open(dirName, 'rb') as f:
            rwses[exps[exp_i]+densities[density_i]] = dill.load(f, ignore=True)
# %%
fig, axs = plt.subplots(1, 2, figsize=(10,5))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

considered_states = ['vel_m','lat_vel']

exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)
legends = ['s=1','s=1','s=1','s=1','s=1',]
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('$\dot x_{0}$ RWSW [m] ')
axs[0].set_ylim([0,2.6])
axs[0].yaxis.set_ticks(np.arange(0, 2.6, 0.25))

axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('$\dot y_{0}$ RWSW [$ms^{-1}$] ')
axs[1].set_ylim([0,2.6])
axs[1].yaxis.set_ticks(np.arange(0, 2.6, 0.25))

for key in range(2):
    for exp_name in exp_names:
        axs[key].plot(np.arange(0,2.1,0.1), rwses[exp_name][considered_states[key]])

    axs[key].grid(axis='y')
axs[0].legend(legends)
# %%

""" scene evolution plots
"""
def choose_traj(actions, prob_mlon, prob_mlat):
    """Returns the index of the chosen traj
        Criteria:
        - yield and merge vehicle jerk
        - likelihood of merge vehicle actions based on the [trajectory] distribution.
        Note: cost componetns are normalized.
    """
    discount_factor = 0.9
    gamma = np.power(discount_factor, np.array(range(0,20)))

    jerk_m_long = (actions[:,1:,0]-actions[:,:-1,0])**2
    jerk_m_lat = (actions[:,1:,1]-actions[:,:-1,1])**2
    jerk_y = (actions[:,1:,2]-actions[:,:-1,2])**2
    jerk_weight = 1/500
    total_cost = jerk_weight*jerk_m_long + jerk_weight*jerk_m_lat + jerk_weight*jerk_y
    likelihoods = np.sum(prob_mlon, axis=1).flatten()+np.sum(prob_mlat, axis=1)[:,:].flatten()
    discounted_cost = np.sum(total_cost*gamma, axis=1)+1/likelihoods
    chosen_traj = np.where(discounted_cost==min(discounted_cost))
    return int(chosen_traj[0])
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(1037)
# st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2765)
st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2215)
pred_h = 2
m_speed_indx = eval_obj.gen_model.indx_m['vel']
m_pc_indx = eval_obj.gen_model.indx_m['pc']
y_speed_indx = eval_obj.gen_model.indx_y['vel']
f_speed_indx = eval_obj.gen_model.indx_f['vel']
fadj_speed_indx = eval_obj.gen_model.indx_fadj['vel']
y_dx_indx = eval_obj.gen_model.indx_y['dx']
f_dx_indx = eval_obj.gen_model.indx_f['dx']
fadj_dx_indx = eval_obj.gen_model.indx_fadj['dx']
fig, axs = plt.subplots(3, 5, figsize=(30,12))
fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
time_frame = 0
for time_step in [19, 29, 39]:
# for time_step in [19]:
    st_i, cond_i, bc_der_i, history_i, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                    cond_seq,
                                                    st_arr,
                                                    targ_arr,
                                                    current_step=time_step,
                                                    pred_h=pred_h)


    actions, prob_mlon, prob_mlat = eval_obj.policy.get_actions([st_i.copy(), cond_i.copy()], bc_der_i, traj_n=100, pred_h=pred_h)
    best_traj = choose_traj(actions, prob_mlon, prob_mlat)
    titles = [
            'Vehicle $v_{0}$,'+
                ' $\dot x_{0}:$'+str(round(st_arr[time_step, m_speed_indx], 1))+'$ms^{-1},$',
            'Vehicle $v_{0}$,'+
                ' $\Delta y_{0}$:'+str(round(st_arr[time_step, m_pc_indx], 1))+'$m$',
            'Vehicle $v_{1}$,'+
                ' $\dot x_{1}:$'+str(round(st_arr[time_step, y_speed_indx], 1))+'$ms^{-1}$,'+
                ' $\Delta x_{1}$:'+str(round(st_arr[time_step, y_dx_indx], 1))+'$m$',
            'Vehicle $v_{2}$,'+
                ' $\dot x_{2}:$'+str(round(st_arr[time_step, fadj_speed_indx], 1))+'$ms^{-1}$,'+
                ' $\Delta x_{2}$:'+str(round(st_arr[time_step, fadj_dx_indx], 1))+'$m$',
            'Vehicle $v_{3}$,'+
                ' $\dot x_{3}:$'+str(round(st_arr[time_step, f_speed_indx], 1))+'$ms^{-1}$,'+
                ' $\Delta x_{3}$:'+str(round(st_arr[time_step, f_dx_indx], 1))+'$m$'
            ]

    for ax_i in range(5):
        axs[time_frame, ax_i].set_ylim([-2,2])
        axs[time_frame, ax_i].set_xlim([-1.9,2.2])
        axs[time_frame, ax_i].spines['right'].set_visible(False)
        axs[time_frame, ax_i].spines['top'].set_visible(False)
        axs[time_frame, ax_i].xaxis.get_major_ticks()[1].label1.set_visible(False)
        axs[time_frame, ax_i].xaxis.get_major_ticks()[2].label1.set_visible(False)
        axs[time_frame, ax_i].xaxis.get_major_ticks()[3].label1.set_visible(False)
        axs[time_frame, ax_i].grid()
        if ax_i == 1:
            axs[time_frame, ax_i].set_ylabel('Lateral action [$ms^{-1}$]')
            # axs[time_frame, ax_i].set_ylabel('Lateral action [$ms^{-1}$]', labelpad=-5)
        else:
            axs[time_frame, ax_i].set_ylabel('Longitudinal action [$ms^{-2}$]')
            # axs[time_frame, ax_i].set_ylabel('Longitudinal action [$ms^{-2}$]', labelpad=-5)
        axs[time_frame, ax_i].set_xlabel('Time [s]')
        #
        # if time_frame!=2:
        #     # axs[time_frame, ax_i].set_yticks([])
        #     axs[time_frame, ax_i].set_xticklabels([])

    for act_n in range(5):

        axs[time_frame, act_n].fill_between([-1.9,0],[-3,-3], [3,3], color='lightgrey')
        axs[time_frame, act_n].title.set_text(titles[act_n])
        axs[time_frame, act_n].plot(np.arange(0, pred_h+0.1, 0.1), targ_i[:, act_n], color='red', linestyle='--')
        axs[time_frame, act_n].plot(np.arange(-1.9, 0.1, 0.1), history_i[:, act_n], color='black', linewidth=2)
        if act_n < 2:
            trajs = actions[:,:,act_n]
            st_dev = np.std(trajs, axis=0)
            avg_traj = np.mean(trajs, axis=0)
            axs[time_frame, act_n].plot(np.arange(0, pred_h+0.1, 0.1), actions[best_traj,:,act_n], color='green', linestyle='--')
            axs[time_frame, act_n].fill_between(np.arange(0, pred_h+0.1, 0.1), avg_traj+st_dev, avg_traj-st_dev, color='orange', alpha=0.3)
            # axs[time_frame, act_n].plot(np.arange(0, pred_h+0.1, 0.1), avg_traj, color='purple')

        else:
            for trj in range(50):
                axs[time_frame, act_n].plot(np.arange(0, pred_h+0.1, 0.1), actions[trj,:,act_n], color='grey', linewidth=0.3)
                # axs[time_frame, act_n].plot(np.arange(0, pred_h+0.1, 0.1), actions[trj,:,act_n], color='grey', linewidth=0.3, alpha=0.3)
    time_frame += 1
# plt.savefig("scene_evolution.png", dpi=200)
# %%


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
