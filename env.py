from graphics import Viewer
import math

class Env():
    RIGHT_LANE_REWARD = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds."""

    LANE_CHANGE_REWARD = -0.1
    """The reward received at each lane change decision."""

    COLLISION_REWARD = -5
    """The reward received when two cars collide."""

    def __init__(self, lane_count):

        self.lane_count = lane_count
        self.lane_width = 3.7 #m
        self.viewer = None
        self.observation_history = []
        self.vehicles = {} # cars indexed by their id
        self.ego = None
        self.max_speed = 20 # m/s


    def reset(self):
        return self._get_observation()

    def render(self, PAUSE_CONTROL=None):
        if self.viewer is None:
            self.viewer = Viewer(self)
            self.viewer.PAUSE_CONTROL = PAUSE_CONTROL

        self.viewer.plotlive()

        if self.viewer.PAUSE_CONTROL == 'on' and \
                        self.ego.remaining_budget == self.ego.available_budget:
            input("Press Enter to continue...")


    def record(self, env_object=None, attribute=None):
        if self.viewer.record_var and not env_object:
            """ record """
            var = self.viewer.record_var[0][self.viewer.record_var[1]]
            self.viewer.record(var)

        elif not self.viewer.record_var and not env_object:
            """ recording not needed """
            return
        else:
            if not self.viewer.record_var:
                """ recording needed - initialize recording """
                if attribute not in env_object.__dict__:
                    raise NotImplementedError("The variable is not an environment attribute")

                self.viewer.record_var = [env_object.__dict__, attribute]


    def action_steps(self, obs):
        while self.ego.remaining_budget > 0:
            if self.viewer is not None:
                self.render(PAUSE_CONTROL=None)
                self.record()

            self.ego.step()
            for car_id in self.vehicles:
                self.vehicles[car_id].step()

            obs = self._get_observation()
            terminal = self.check_terminal(obs)
            if terminal:
                return obs, terminal

            self.ego.remaining_budget -= 0.1

        return obs, terminal

    def decision_step(self, obs, decision):
        self.ego.remaining_budget = self.ego.available_budget
        self.ego.decision = self.ego.ALL_OPTIONS[decision]

        obs, terminal = self.action_steps(obs)
        reward = self._next_reward(obs, terminal)

        return obs, reward, terminal


    def detect_lc(self, car_id):
        """
        Note: last obs in observation_history is x1 time-step behind
        """

        if len(self.observation_history) != 0:
            lane_id_current = self.ego.lane_id
            lane_id_previous = self.observation_history[-1][car_id]['lane_id']

            if lane_id_previous == lane_id_current:
                return False

            else:
                if lane_id_previous > lane_id_current:
                    return 'right'
                else:
                    return 'left'

    def neighbour_idendifier(self, detect_lc):
        left = ['fl', 'bl']
        center = ['f', 'b']
        right = ['fr', 'br']

        if detect_lc == 'left':
            for i in range(2):
                if center[i] in self.vehicles:
                    self.vehicles[right[i]] = self.vehicles.pop(center[i])
                if left[i] in self.vehicles:
                    self.vehicles[center[i]] = self.vehicles.pop(left[i])
        else:
            for i in range(2):
                if center[i] in self.vehicles:
                    self.vehicles[left[i]] = self.vehicles.pop(center[i])
                if right[i] in self.vehicles:
                    self.vehicles[center[i]] = self.vehicles.pop(right[i])

    def _update_vehicle_ids(self, obs, car_id):
        """
        Update id of other cars as the scene evolves.
        Right now it is assumed other cars only perform lane keeping
        """
        if car_id == 'ego':
            detect_lc = self.detect_lc(car_id) # left, right, False
            if detect_lc:
                self.neighbour_idendifier(detect_lc)

        else:
            front_orientations = ['fl','fr']
            back_orientations = ['bl','br']
            _car_id = None

            if car_id in  front_orientations and obs['dx'] > 0:
                _car_id = back_orientations[front_orientations.index(car_id)]

            if car_id in  back_orientations and obs['dx'] < 0:
                _car_id = front_orientations[back_orientations.index(car_id)]


            obs['dx'] = abs(obs['dx'])
            if _car_id is not None:
                if _car_id in self.vehicles:
                    del self.vehicles[_car_id]

                self.vehicles[_car_id] = self.vehicles.pop(car_id)
            else:
                _car_id = car_id

            return obs, _car_id

    def _get_observation(self):
        """
        Next observation - used for reward calculation and decision prediction.
        """
        obs_all = {'ego': {}}

        obs_all['ego']['lane_id'] = self.ego.lane_id
        obs_all['ego']['speed_long'] = self.ego.speed_long
        obs_all['ego']['act_long'] = self.ego.act_long

        obs_all['ego']['x'] = self.ego.x
        obs_all['ego']['y'] = self.ego.y

        obs_all['ego']['y_cor'] = self.ego.y_cor
        self._update_vehicle_ids(obs=None, car_id='ego')

        for car_id in list(self.vehicles):
            obs = {}
            obs['dx'] = self.ego.x - self.vehicles[car_id].x
            obs['dy'] = abs(self.ego.y_cor - self.vehicles[car_id].y_cor)
            obs['vxd'] = self.ego.speed_long - self.vehicles[car_id].speed_long
            obs, _car_id = self._update_vehicle_ids(obs, car_id)
            obs['time_headway'] = obs['dx']/self.ego.speed_long

            obs_all[_car_id] = obs

        self.observation_history.append(obs_all)
        return obs_all

    def check_terminal(self, obs,):
        if self.vehicles:
            for car_id in self.vehicles:
                if obs[car_id]['dx'] < 6 and \
                                obs[car_id]['dy'] < 2:

                    return True

        return False

    def _next_reward(self, obs, terminal):
        reward = 0
        speed_cost_base = 15

        if terminal:
            reward += self.COLLISION_REWARD

        if self.ego.speed_long < self.max_speed:
            reward -= abs(self.ego.speed_long - self.max_speed)/self.max_speed

        else:
            if self.ego.act_long >= 0:
                reward -= 0.5

        reward -= (speed_cost_base**abs(self.ego.act_long) - 1) / speed_cost_base**3



        # if self.ego.lane_id == 1:
        #     reward += 0.4

        # if obs['ego']['lane_id'] == 1:
        #     reward += -1
        return reward
