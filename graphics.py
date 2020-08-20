import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
import time
import numpy as np

class Viewer():
    STEP_SIZE = 0.0000001 # for visualisations
    def __init__(self, env):
        self.step_count = 30 # total number of steps
        self.steps = 0 # steps rendered
        self.env = env
        self.fig = plt.figure(figsize=(15, 5))
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                         height_ratios=[2,4],
                         width_ratios=[4,2])

        self.ax_scene = self.fig.add_subplot(spec[0,0])
        self.ax_recorded_variable = self.fig.add_subplot(spec[1,0])
        self.ax_value_bar = self.fig.add_subplot(spec[0,1])

        self.var_recording = []
        self.record_var = None

        self.ego = self.env.ego

        self.road_start = 0 #[m]
        self.road_end = 10000 #[m]
        self.perception_horizon = 70
        self.PAUSE_CONTROL = 'off'
        self.dynamic_objects = [] # update every frame
        self.decision_objects = [] # update everytime decision is made
        self.ego_label = None
        self.value_bar = None

    def set_ego_label(self, env_object=None, attribute=None):
        if self.ego_label is None:
            self.ego_label = [env_object.__dict__, attribute]

    def draw_tree(self, ax):

        for traj in self.ego.traj:
            fig_obj = ax.plot(traj[0], traj[1], '-o', \
                                        markersize=3, alpha=0.2, color='black')

    def draw_car(self, ax, car_id):

        if car_id == 'ego':
            car = self.ego
            color = 'red'
            if self.ego_label:
                label = "{:.1f}".format(self.ego_label[0][self.ego_label[1]])
            else:
                label = 'ego'
        else:
            car = self.env.vehicles[car_id]
            color = 'grey'
            label = car_id

            ellipse = Ellipse((car.x, car.y_cor),
                width=6,
                height=2,
                facecolor='red',
                alpha=0.1)

            fig_obj = ax.add_patch(ellipse)
            self.dynamic_objects.append(fig_obj)

        fig_obj = ax.scatter(car.x, car.y_cor, s=100, marker=">", color=color)
        self.dynamic_objects.append(fig_obj)
        fig_obj = ax.annotate(label,
                     (car.x, car.y_cor),
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center')
        self.dynamic_objects.append(fig_obj)

    def draw_scene(self):
        """
        This remains fixed
        """

        ax = self.ax_scene
        ax.hlines(self.env.lane_width * self.env.lane_count, self.road_start,
                                    self.road_end, colors='k', linestyles='solid')

        ax.hlines(self.env.lane_width, self.road_start, self.road_end,
                                                        colors='k', linestyles='--')

        ax.hlines(0, self.road_start, self.road_end, colors='k', linestyles='solid')
        ax.axes.set_ylim(0, self.env.lane_width * self.env.lane_count)

        ax.set_yticks([])


        ax = self.ax_value_bar
        ax.set_ylabel('Decision counts')
        ax.set_yticks([])
        ax.set_xticks([])

    def draw_recorded_var(self, ax):
        ax.axes.set_xlim(0, len(self.var_recording) * 1.5)
        ax.axes.set_ylim(min(self.var_recording), max(self.var_recording))
        var_len = len(self.var_recording)

        if var_len == 1:# do it once
            ax.axes.grid()
            ax.set_xlabel('time steps')
            ax.set_ylabel(self.record_var[1])

        fig_obj = ax.plot(range(var_len), self.var_recording, color='black')
        self.dynamic_objects.append(fig_obj[0])

    def figure_update(self, ax):

        road_end = self.ego.x + self.perception_horizon
        road_start = self.ego.x - self.perception_horizon
        if road_start < 0:
            road_start = 0

        ax.axes.set_xlim(road_start, road_end)


    def draw_decision_value_bar(self, ax):

        decisions = self.ego.decision_counts['decisions']
        counts =  self.ego.decision_counts['counts']

        poses = np.arange(len(decisions))
        lat_labels = [self.ego.ALL_OPTIONS.get(key) for key in decisions]
        lat_labels = [item[0]+'_'+item[1] for item in lat_labels]


        colors = ['green' if val == max(counts) else 'grey' for val in counts]

        ax.set_xticks(poses)
        ax.set_xticklabels(lat_labels, rotation=90)
        fig_obj = ax.bar(poses, counts, align='center', \
                                                width=0.5, color=colors)
        self.decision_objects.append(fig_obj)


    def record(self, state_variable):
        """
        records a given variable
        """
        self.var_recording.append(state_variable)

    def clear_viewer(self, figure_objects):

        if figure_objects:
            for obj in figure_objects:
                obj.remove()

    def plotlive(self):
        if self.ego.remaining_budget == self.ego.available_budget:
            self.ax_scene.clear()
            self.draw_scene()
            self.draw_tree(self.ax_scene)
            if self.ego.decision_counts:
                self.draw_decision_value_bar(self.ax_value_bar)

        self.clear_viewer(self.dynamic_objects)
        self.dynamic_objects = []

        self.figure_update(self.ax_scene)
        self.draw_car(self.ax_scene, 'ego')

        for car_id in self.env.vehicles:
            self.draw_car(self.ax_scene, car_id)

        if self.var_recording:
            self.draw_recorded_var(self.ax_recorded_variable)

        plt.pause(self.STEP_SIZE)
