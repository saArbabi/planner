from agents import Vehicle, EgoVehicle
from env import Env
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
env = Env(2)
env.ego = EgoVehicle(env, x=30, vel=10, lane_id=1)
env.vehicles['bl'] = Vehicle(x=10, vel=9, lane_id=2)
env.vehicles['bl'].act_long = 1
# env.vehicles['f'] = Vehicle(x=30, vel=10, lane_id=1)

# env.vehicles['f'] = Vehicle(x=60, vel=9, lane_id=1)
 
obs = env.reset()
# env.record (env_object=env.ego, attribute='vel')
# env.record (env_object=env.ego, attribute='vel')
# env.record(env_object=env.vehicles['f'], attribute='desired_headway')

# env.render(PAUSE_CONTROL='on')
env.render(PAUSE_CONTROL='on')
# env.render(PAUSE_CONTROL='off')

env.record(env_object=env.ego, attribute='vel')
env.viewer.set_ego_label(env_object=env.ego, attribute='vel')
# x = np.array([x/10 for x in range(30)])
# y = eval('(5**x - 1)/125')
# plt.plot(x, y)
# env.record(obs, env_object=env.vehicles['f'], attribute='dx')
decision = 2
for i in range(200):
    t0 = time.time()

    decision = env.ego.make_decision(obs)
    obs, reward, terminal = env.decision_step(obs, decision)
    if terminal:
        break

    # env.ego.reward = reward
    print(time.time() - t0)
    # t_start = time.time()
    # t_end = time.time()
    # print(t_end - t_start)

    # print(obs)
