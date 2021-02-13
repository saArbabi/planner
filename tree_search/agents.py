import matplotlib.pyplot as plt
import time
from mcts import MCTSDPW
# from matplotlib import style


class Vehicle(object):
    step_size = 0.1 # s update rate - used for dynamics

    def __init__(self, x, vel, lane_id):
        self.x = x
        self.vel = vel # longitudinal speed [m/s]
        self.desired_headway = None

        self.act_long = 0 # longitudinal acceleration [m/s^2]
        self.act_lat = 0 # lateral speed
        self.lane_id = lane_id # inner most right is 1
        self.y = 0

        if self.lane_id != 1:
            self.y_cor = 1.85 * self.lane_id * 1.5
        else:
            self.y_cor = 1.85


    def act(self):
        """
        :param high-level decision of the car
        """
        pass

    def step(self):

        self.act()

        self.x = self.x + self.vel * self.step_size \
                                    + 0.5 * self.act_long * self.step_size **2

        y_delta = self.act_lat*self.step_size
        self.y += y_delta
        self.vel = self.vel + self.act_long * self.step_size
        self.y_cor += y_delta

        if self.y >= 1.85 and self.act_lat > 0:
            self.lane_id += 1
            self.y = -self.y + y_delta

        if self.y <= -1.85 and self.act_lat < 0:
            self.lane_id -= 1
            self.y = -self.y + y_delta

class EgoVehicle(Vehicle):
    ALL_OPTIONS = {
                0: ['LK', 'UP'],
                1: ['LK', 'DOWN'],
                2: ['LK', 'IDLE'],
                3: ['LCL', 'UP'],
                4: ['LCL', 'DOWN'],
                5: ['LCL', 'IDLE'],
                6: ['LCR', 'UP'],
                7: ['LCR', 'DOWN'],
                8: ['LCR', 'IDLE']
                }

    def __init__(self, env, x, vel, lane_id):
        super(EgoVehicle, self).__init__(x, vel, lane_id)

        self.env = env
        self.controller = Controller(env, self)
        self.headway_change = 0.1
        self.reward = 0
        self.decision = None
        self.remaining_budget = 1 #[s]
        self.available_budget = 1 #[s]
        self.traj = []
        self.decision_counts = None
        self.planner = MCTSDPW(env, self.remaining_budget)



    def act(self):
        if self.decision[0] == 'LK':
            self.controller.lk_control()

        elif self.decision[0] == 'LCL' or self.decision[0] == 'LCR':
            self.controller.lc_control(self.decision[0])

        self.act_long = self.controller.speed_control(self.decision[1])



    def initialize_timeheadway(self, obs, vehicle, car_id):
        vehicle.desired_headway = obs[car_id]['time_headway']

    def set_timeheadway(self, obs, car_id):

        if car_id in self.env.vehicles:
            vehicle = self.env.vehicles[car_id]
            if vehicle.desired_headway == None:
                self.initialize_timeheadway(obs, vehicle, car_id)

            else:
                if self.decision[1] == 'UP':
                    vehicle.desired_headway += self.headway_change * (0 - obs[car_id]['time_headway'])
                elif self.decision[1] == 'DOWN':
                    vehicle.desired_headway += self.headway_change * (2 - obs[car_id]['time_headway'])

    def make_decision(self, obs):
        """
        :return: a high-level policy
        """
        decision = self.planner.plan(self.env, obs)
        return decision
        #TODO - this must come from mcts

class Controller(object):
    def __init__(self, env, ego):
        self.env = env
        self.ego = ego
        self.desired_headway = None
        self.desired_speed = self.ego.vel #[m/s]
        self.max_speed = 20 #[m/s]
        self.min_speed = 10 #[m/s]



    def clip_action(self, action, max_value):
        if action < -max_value:
            return -max_value
        elif action > max_value:
            return max_value
        else:
            return action

    def gap_control(self, obs, car_id):
        """
        Adjust time_headway with a true or hypothetical vehicle.
        Chosen vehicle depends on:
            Currently: decision
            Future: proximity too
        """

        if self.ego.remaining_budget == self.ego.available_budget:
            self.ego.set_timeheadway(obs, car_id)

        acc = obs[car_id]['time_headway'] - self.env.vehicles[car_id].desired_headway
        self.ego.act_long = self.clip_action(acc, max_value=3)


    def lk_control(self):
        self.ego.act_lat = -self.ego.y

    def speed_control(self, acceleration_change):
        gain = 0.1
        acc = self.ego.act_long

        if acceleration_change == 'UP':
            if self.ego.act_long < 0:
                return 0
            acc += gain * (3-self.ego.act_long)
        elif acceleration_change == 'DOWN':
            if self.ego.act_long > 0:
                return 0
            acc += gain * (-3-self.ego.act_long)
        else:
            return acc


        return self.clip_action(acc, max_value=3)

        # if self.ego.remaining_budget == self.ego.available_budget:
        #
        #     if acceleration_change == 'UP':
        #         self.desired_speed += 5
        #     if acceleration_change == 'DOWN':
        #         self.desired_speed -= 5
        #
        # acc = self.desired_speed - self.ego.vel
        # self.ego.act_long = self.clip_action(acc, max_value=3)

        # if acceleration_change == 'UP':
        #     self.ego.act_long = 1
        # if acceleration_change == 'DOWN':
        #     self.ego.act_long = -1




    def lc_control(self, decision):

        if decision == 'LCL':
            if self.ego.lane_id == 1:
                self.ego.act_lat = abs(self.ego.y) + 0.1 # lateral speed
            else:
                self.ego.act_lat = -self.ego.y

        if decision == 'LCR':
            if self.ego.lane_id == 2:
                self.ego.act_lat = self.ego.y - 0.1 # lateral speed
            else:
                self.ego.act_lat = -self.ego.y
