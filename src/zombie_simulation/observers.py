from __future__ import annotations

import builtins
import io
import math
import os
import time
import tkinter as tk
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime

import keras
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pygame
import scipy
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.callbacks import LambdaCallback
from matplotlib import animation, colors, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from numpy.fft import fft, fft2, fftshift
from plotly.subplots import make_subplots
from population import Population
from scipy import stats
from scipy.interpolate import interp1d
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from states import HealthState
from tensorboard import default, program
from webdriver_manager.chrome import ChromeDriverManager

# Observer Pattern

class Observer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def display_observation(self) -> None:
        pass


class SimulationObserver(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.statistics = []
        self.grid = self.subject.school.grid
        self.agent_list = self.subject.agent_list

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

    def update(self) -> None:
        statistics =  {
            "num_healthy": self.subject.num_healthy,
            "num_infected": self.subject.num_infected,
            "num_zombie": self.subject.num_zombie,
            "num_dead": self.subject.num_dead,
            "population_size": self.subject.population_size,
            "infection_probability": self.subject.infection_probability,
            "turning_probability": self.subject.turning_probability,
            "death_probability": self.subject.death_probability,
            "migration_probability": self.subject.migration_probability,
        }
        self.statistics.append(deepcopy(statistics))
        self.grid = deepcopy(self.subject.school.grid)
        self.agent_list = deepcopy(self.subject.agent_list)
        

    def display_observation(self, format="statistics"):
        if format == "statistics":
            self.print_statistics_text()
        elif format == "grid":
            self.print_grid_text()
        elif format == "bar":
            self.print_bar_graph()
        elif format == "scatter":
            self.print_scatter_graph()
        elif format == "table":
            self.print_table_graph()

    def print_statistics_text(self):
        population_size = self.statistics[-1]["population_size"]
        num_healthy = self.statistics[-1]["num_healthy"]
        num_infected = self.statistics[-1]["num_infected"]
        num_zombie = self.statistics[-1]["num_zombie"]
        num_dead = self.statistics[-1]["num_dead"]
        healthy_percentage = num_healthy / (population_size + 1e-10)
        infected_percentage = num_infected / (population_size + 1e-10)
        zombie_percentage = num_zombie / (population_size + 1e-10)
        dead_percentage = num_dead / (population_size + 1e-10)
        infected_rate = num_infected / (num_healthy + 1e-10)
        turning_rate = num_zombie / (num_infected + 1e-10)
        death_rate = num_dead / (num_zombie + 1e-10)
        infection_probability = self.statistics[-1]["infection_probability"]
        turning_probability = self.statistics[-1]["turning_probability"]
        death_probability = self.statistics[-1]["death_probability"]
        migration_probability = self.statistics[-1]["migration_probability"]
        print("Population Statistics:")
        print(f"Population Size: {population_size}")
        print(f"Healthy: {num_healthy} ({healthy_percentage:.2%})")
        print(f"Infected: {num_infected} ({infected_percentage:.2%})")
        print(f"Zombie: {num_zombie} ({zombie_percentage:.2%})")
        print(f"Dead: {num_dead} ({dead_percentage:.2%})")
        print(f"Infection Probability: {infection_probability:.2%} -> Infected Rate: {infected_rate:.2%}")
        print(f"Turning Probability: {turning_probability:.2%} -> Turning Rate: {turning_rate:.2%}")
        print(f"Death Probability: {death_probability:.2%} -> Death Rate: {death_rate:.2%}")
        print(f"Migration Probability: {migration_probability:.2%}")
        
        # The mean can be used to calculate the average number of zombies that appear in a specific area over time. This can be useful for predicting the rate of zombie infection and determining the necessary resources needed to survive.
        mean = np.mean([d["num_zombie"] for d in self.statistics])
        # The median can be used to determine the middle value in a set of data. In a zombie apocalypse simulation, the median can be used to determine the number of days it takes for a specific area to become overrun with zombies.
        median = np.median([d["num_zombie"] for d in self.statistics])
        # The mode can be used to determine the most common value in a set of data. In a zombie apocalypse simulation, the mode can be used to determine the most common type of zombie encountered or the most effective weapon to use against them.
        mode = stats.mode([d["num_zombie"] for d in self.statistics], keepdims=True)[0][0]
        # The standard deviation can be used to determine how spread out a set of data is. In a zombie apocalypse simulation, the standard deviation can be used to determine the level of unpredictability in zombie behavior or the effectiveness of certain survival strategies.
        std = np.std([d["num_zombie"] for d in self.statistics])
        print(f"Mean of Number of Zombie: {mean}")
        print(f"Median of Number of Zombie: {median}")
        print(f"Mode of Number of Zombie: {mode}")
        print(f"Standard Deviation of Number of Zombie: {std}")
        print()

    """
    # may format the output in the subject and print it directly here
    
    def print_text_statistics(self):
        print("Population Statistics:")
        for key, value in self.statistics.items():
            print(f"{key}: {value}")
            
    # subject notify method push info to update method of observer
    # but the subject don't know what info observer want to display
    # or let observer pull info from subject using get method of subject
    # but observer need to get info from subject one by one
    """
    
    def print_grid_text(self):
        print("Print School:")
        state_symbols = {
            None: " ",
            HealthState.HEALTHY: "H",
            HealthState.INFECTED: "I",
            HealthState.ZOMBIE: "Z",
            HealthState.DEAD: "D",
        }

        for row in self.grid:
            for cell in row:
                try:
                    print(state_symbols[cell.health_state], end=" ")
                except AttributeError:
                    print(state_symbols[cell], end=" ")
            print()
        print()


    def print_bar_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.set_title("Bar Chart")
        ax.set_ylim(0, self.statistics[0]["population_size"] + 1)

        # Use common state_colors
        cell_states = [individual.health_state for individual in self.agent_list]
        counts = {state: cell_states.count(state) for state in HealthState}
        ax.bar(
            np.asarray(HealthState.value_list()),
            list(counts.values()),
            tick_label=HealthState.name_list(),
            label=HealthState.name_list(),
            color=[self.state_colors[state] for state in HealthState]
        )

        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_scatter_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.set_title("Scatter Chart")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)

        x = np.array([individual.location[0] for individual in self.agent_list])
        y = np.array([individual.location[1] for individual in self.agent_list])
        cell_states_value = np.array([individual.health_state.value for individual in self.agent_list])

        # Use common cmap
        ax.scatter(x, y, c=cell_states_value, cmap=self.cmap)

        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_table_graph(self):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Table Chart")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')

        # Initialize table with common state_colors
        cell_states = [["" for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        for j, individual in enumerate(self.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        table = ax.table(cellText=np.array(cell_states), loc="center", bbox=Bbox.from_bounds(0, 0, 1, 1))

        # Adjust cell properties using common state_colors
        for key, cell in table.get_celld().items():
            cell_state = cell_states[key[0]][key[1]]
            cell.set_facecolor(self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white")
            cell.get_text().set_text(cell_state)
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
        
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


class SimulationAnimator(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_history = []

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

    def update(self) -> None:
        self.agent_history.append(deepcopy(self.subject.agent_list))

    def display_observation(self, format="bar"):
        if format == "bar":
            self.print_bar_animation()
        elif format == "scatter":
            self.print_scatter_animation()
        elif format == "table":
            self.print_table_animation()

    def print_bar_animation(self):
        counts = []
        for agent_list in self.agent_history:
            cell_states = [individual.health_state for individual in agent_list]
            counts.append([cell_states.count(state) for state in HealthState])

        self.bar_chart_animation(np.array(HealthState.value_list()), np.array(counts), HealthState.name_list())

    def bar_chart_animation(self, x, y, ticks):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Bar Chart Animation")
        ax.set_ylim(0, max(map(max, y)) + 1)

        bars = ax.bar(x, y[0], tick_label=ticks, label=HealthState.name_list(), color=self.state_colors.values())
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        def update(i):
            for j, bar in enumerate(bars):
                bar.set_height(y[i][j])
            text_box.set_text(f"Time Step: {i+1}")

        anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000, repeat=False)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_scatter_animation(self):
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[self.subject.school.size - individual.location[0] - 1 for individual in agent_list] for agent_list in self.agent_history]
        cell_states_value = [[individual.health_state.value for individual in agent_list] for agent_list in self.agent_history]

        self.scatter_chart_animation(x, y, cell_states_value)

    def scatter_chart_animation(self, x, y, cell_states_value):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Scatter Chart Animation")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)

        sc = ax.scatter(x[0], y[0], c=cell_states_value[0], cmap=self.cmap)
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        def animate(i):
            sc.set_offsets(np.c_[x[i], y[i]])
            sc.set_array(cell_states_value[i])
            text_box.set_text(f"Time Step: {i+1}")

        anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=1000, repeat=False)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_table_animation(self):
        cell_states_name = [[individual.health_state for individual in agent_list] for agent_list in self.agent_history]
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[individual.location[1] for individual in agent_list] for agent_list in self.agent_history]

        # Build the grid
        cell_states = []
        for i in range(len(cell_states_name)):
            grid = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
            for j, individual in enumerate(cell_states_name[i]):
                grid[x[i][j]][y[i][j]] = individual.name
            cell_states.append(grid)

        self.table_animation(cell_states)

    def table_animation(self, cell_states):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Table Animation")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')

        # Initialize table
        table = ax.table(cellText=cell_states[0], loc="center", bbox=Bbox.from_bounds(0, 0, 1, 1))
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        # Adjust cell properties for centering text
        for key, cell in table.get_celld().items():
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')

        def animate(i):
            for row_num, row in enumerate(cell_states[i]):
                for col_num, cell_value in enumerate(row):
                    cell_color = self.state_colors.get(HealthState[cell_value], "white") if cell_value else "white"
                    table[row_num, col_num].set_facecolor(cell_color)
                    table[row_num, col_num].get_text().set_text(cell_value)
            text_box.set_text(f"Time Step: {i+1}")
            return table, text_box

        anim = animation.FuncAnimation(fig, animate, frames=len(cell_states), interval=1000, repeat=False, blit=True)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


class PlotlyAnimator(Observer):
    def __init__(self, population: Population):
        self.subject = population
        self.subject.attach_observer(self)
        self.data_history = []

    def update(self):
        current_state = self.capture_current_state()
        self.data_history.append(current_state)

    def capture_current_state(self):
        data = [{'x': ind.location[0], 'y': ind.location[1], 'z': 0, 'state': ind.health_state.name} for ind in self.subject.agent_list]
        return pd.DataFrame(data)

    def display_observation(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Scatter Plot", "Heatmap", "Time Series", "3D Scatter Plot"),
                            specs=[[{"type": "scatter"}, {"type": "heatmap"}], [{"type": "scatter"}, {"type": "scatter3d"}]])

        self.add_scatter_plot(fig, row=1, col=1)
        self.add_heatmap(fig, row=1, col=2)
        self.add_time_series(fig, row=2, col=1)
        self.add_3d_scatter(fig, row=2, col=2)

        fig.update_layout(height=800, width=1200, title_text="Zombie Apocalypse Simulation",
                          legend_title="Health States", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.show()

    def add_scatter_plot(self, fig, row, col):
        scatter_data = self.data_history[-1]
        scatter_plot = px.scatter(scatter_data, x="x", y="y", color="state")
        for trace in scatter_plot.data:
            fig.add_trace(trace, row=row, col=col)

    def add_heatmap(self, fig, row, col):
        heatmap_data = self.data_history[-1].pivot_table(index='y', columns='x', aggfunc='size', fill_value=0)
        fig.add_trace(go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'), row=row, col=col)

    def add_time_series(self, fig, row, col):
        time_series_data = self.prepare_time_series_data()
        time_series_plot = px.line(time_series_data, x="time_step", y="counts", color='state')
        for trace in time_series_plot.data:
            fig.add_trace(trace, row=row, col=col)

    def add_3d_scatter(self, fig, row, col):
        scatter_data = self.data_history[-1]
        scatter_3d = px.scatter_3d(scatter_data, x="x", y="y", z="z", color="state")
        for trace in scatter_3d.data:
            fig.add_trace(trace, row=row, col=col)

    def prepare_time_series_data(self):
        all_states = ['HEALTHY', 'INFECTED', 'ZOMBIE', 'DEAD']
        all_combinations = pd.MultiIndex.from_product([range(len(self.data_history)), all_states], names=['time_step', 'state']).to_frame(index=False)

        time_series_data = pd.concat([data['state'].value_counts().rename_axis('state').reset_index(name='counts').assign(time_step=index) for index, data in enumerate(self.data_history)], ignore_index=True)
        
        return pd.merge(all_combinations, time_series_data, on=['time_step', 'state'], how='left').fillna(0)


class MatplotlibAnimator(Observer):
    def __init__(self, population: Population, plot_order=["bar", "scatter", "table"]):
        """Initialize the animator with customizable plot order."""
        self.subject = population
        self.subject.attach_observer(self)
        self.plot_order = plot_order

        # Initialize matplotlib figure with three subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)

        # Initialize common elements for the plots
        self.init_common_elements()

        # Setup each subplot based on the specified order
        for i, plot_type in enumerate(self.plot_order):
            if plot_type == "bar":
                self.setup_bar_chart(self.axes[i])
            elif plot_type == "scatter":
                self.setup_scatter_plot(self.axes[i])
            elif plot_type == "table":
                self.setup_table(self.axes[i])

    def init_common_elements(self):
        # Common elements initialization
        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

    # Setup methods for each plot type
    def setup_bar_chart(self, ax):
        ax.set_title("Bar Chart")
        ax.set_ylim(0, len(self.subject.agent_list) + 1)
        self.setup_initial_bar_state(ax)

    def setup_scatter_plot(self, ax):
        ax.set_title("Scatter Plot")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)
        self.setup_initial_scatter_state(ax)

    def setup_table(self, ax):
        ax.set_title("Table")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')
        self.setup_initial_table_state(ax)

    # Methods for setting up initial states of each plot type
    def setup_initial_bar_state(self, ax):
        counts = [self.cell_states.count(state) for state in list(HealthState)]
        self.bars = ax.bar(np.array(HealthState.value_list()), counts, tick_label=HealthState.name_list(), 
                                label=HealthState.name_list(), color=[self.state_colors[state] for state in HealthState])
        self.bar_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def setup_initial_scatter_state(self, ax):
        transformed_x_coords = [y for y in self.cell_y_coords]
        transformed_y_coords = [self.subject.school.size - x - 1 for x in self.cell_x_coords]

        self.scatter = ax.scatter(transformed_x_coords, transformed_y_coords, 
                                c=self.cell_states_value, cmap=self.cmap)
        self.scatter_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def setup_initial_table_state(self, ax):
        cell_states = [["" for _ in range(self.subject.school.size)] 
                        for _ in range(self.subject.school.size)]
        for j, individual in enumerate(self.subject.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        self.table = ax.table(cellText=np.array(cell_states), loc="center", bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0))
        self.table_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        # Adjust cell properties for centering text
        for key, cell in self.table.get_celld().items():
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')

        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                cell_state = cell_states[i][j]
                color = self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white"
                self.table[i, j].set_facecolor(color)
                self.table[i, j].get_text().set_text(cell_state)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def display_observation(self):
        plt.show()

    def update(self):
        # Update the elements common to all plots
        self.update_common_elements()

        # Update each subplot based on its type
        for i, plot_type in enumerate(self.plot_order):
            if plot_type == "bar":
                self.update_bar_chart(self.axes[i])
            elif plot_type == "scatter":
                self.update_scatter_plot(self.axes[i])
            elif plot_type == "table":
                self.update_table(self.axes[i])

    def update_common_elements(self):
        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]

    # Update methods for each plot type
    def update_bar_chart(self, ax):
        counts = [self.cell_states.count(state) for state in HealthState]
        for bar, count in zip(self.bars, counts):
            bar.set_height(count)
        self.bar_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)

    def update_scatter_plot(self, ax):
        transformed_x_coords = [self.cell_y_coords[i] for i in range(len(self.cell_x_coords))]
        transformed_y_coords = [self.subject.school.size - self.cell_x_coords[i] - 1 for i in range(len(self.cell_y_coords))]

        self.scatter.set_offsets(np.c_[transformed_x_coords, transformed_y_coords])
        self.scatter.set_array(np.array(self.cell_states_value))
        self.scatter_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)

    def update_table(self, ax):
        cell_states = [["" for _ in range(self.subject.school.size)] 
                        for _ in range(self.subject.school.size)]
        for j, individual in enumerate(self.subject.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        for (i, j), cell in np.ndenumerate(cell_states):
            cell_state = cell_states[i][j]
            color = self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white"
            self.table[i, j].set_facecolor(color)
            self.table[i, j].get_text().set_text(cell_state)
        self.table_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)


class TkinterObserver(Observer):
    def __init__(self, population, grid_size=300, cell_size=30):
        self.subject = population
        self.subject.attach_observer(self)

        # Define grid and cell sizes
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_cells = self.subject.school.size

        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title("Zombie Apocalypse Simulation")

        # Canvas for the simulation grid
        self.grid_canvas = tk.Canvas(self.root, width=self.grid_size, height=self.grid_size)
        self.grid_canvas.pack(side=tk.LEFT)

        # Frame for Matplotlib plots
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup common elements and plots
        self.init_common_elements()
        self.setup_plots()

        self.update()  # Initial update

    def init_common_elements(self):
        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))

    def setup_plots(self):
        self.figures = {
            'bar': Figure(figsize=(7, 7), constrained_layout=True),
            'scatter': Figure(figsize=(7, 7), constrained_layout=True),
            'table': Figure(figsize=(7, 7), constrained_layout=True)
        }

        # Initial setup for each plot
        self.setup_initial_bar_state(self.figures['bar'].add_subplot(111))
        self.setup_initial_scatter_state(self.figures['scatter'].add_subplot(111))
        self.setup_initial_table_state(self.figures['table'].add_subplot(111))

        # Using grid layout manager instead of pack
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_rowconfigure(2, weight=1)

        self.canvases = {}
        for i, (plot_type, fig) in enumerate(self.figures.items()):
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvases[plot_type] = canvas
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=i, column=0, sticky="nsew")

        # Ensure the grid_canvas is positioned correctly
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update(self):
        self.draw_grid()
        self.update_plots()
        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.5)

    def draw_grid(self):
        self.grid_canvas.delete("all")
        for individual in self.subject.agent_list:
            x, y = individual.location
            canvas_x = y
            canvas_y = x
            x1, y1 = canvas_x * self.cell_size, canvas_y * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            color = self.get_color(individual.health_state)
            self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def update_plots(self):
        self.update_bar_chart()
        self.update_scatter_plot()
        self.update_table_plot()

    # Bar chart setup and update
    def setup_initial_bar_state(self, ax):
        ax.set_title("Bar Chart")
        ax.set_ylim(0, len(self.subject.agent_list) + 1)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_bar_chart(self):
        ax = self.figures['bar'].gca()
        ax.clear()
        self.setup_initial_bar_state(ax)
        cell_states = [individual.health_state for individual in self.subject.agent_list]
        counts = {state: cell_states.count(state) for state in HealthState}
        heights = np.array([counts.get(state, 0) for state in HealthState])
        ax.bar(np.arange(len(HealthState)), heights, tick_label=[state.name for state in HealthState], color=[self.state_colors[state] for state in HealthState])
        self.canvases['bar'].draw()

    # Scatter plot setup and update
    def setup_initial_scatter_state(self, ax):
        ax.set_title("Scatter Plot")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_scatter_plot(self):
        ax = self.figures['scatter'].gca()
        ax.clear()
        self.setup_initial_scatter_state(ax)
        x = [individual.location[1] for individual in self.subject.agent_list]
        y = [individual.location[0] for individual in self.subject.agent_list]
        cell_states_value = [individual.health_state.value for individual in self.subject.agent_list]
        ax.scatter(x, y, c=cell_states_value, cmap=self.cmap)
        self.canvases['scatter'].draw()

    # Table plot setup and update
    def setup_initial_table_state(self, ax):
        ax.set_title("Table")
        ax.axis('tight')
        ax.axis('off')
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_table_plot(self):
        ax = self.figures['table'].gca()
        ax.clear()
        self.setup_initial_table_state(ax)
        cell_states = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
        for individual in self.subject.agent_list:
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name
        table = ax.table(cellText=cell_states, loc="center")
        for key, cell in table.get_celld().items():
            cell_state = cell_states[key[0]][key[1]]
            cell.set_facecolor(self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white")
            cell.get_text().set_text(cell_state)
        self.canvases['table'].draw()

    def get_color(self, health_state):
        rgb_color = self.state_colors.get(health_state, (1, 1, 1))  # Default white
        return f"#{int(rgb_color[0]*255):02x}{int(rgb_color[1]*255):02x}{int(rgb_color[2]*255):02x}"

    def display_observation(self):
        self.root.mainloop()


class PredictionObserver(Observer):
    def __init__(self, population: Population, data_dir="./simulation_data") -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.data_dir = data_dir
        self.load_previous_data()
        self.current_grid_history = []
        self.log_dir = "./logs/fit" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/image")

    def load_previous_data(self):
        self.grid_history_path = os.path.join(self.data_dir, "grid_history.npz")
        if os.path.exists(self.grid_history_path):
            with np.load(self.grid_history_path) as data:
                self.past_grid_history = data["grid_history"].tolist()
        else:
            self.past_grid_history = []
        print(f"Loaded {len(self.past_grid_history)} grid states from previous simulation.")

    def update(self) -> None:
        current_grid_state = self.capture_grid_state()
        self.current_grid_history.append(current_grid_state)

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                individual = self.subject.school.get_individual((i, j))
                grid_state[i, j] = individual.health_state.value if individual else 0
        return grid_state

    def prepare_data(self, N):
        combined_grid_history = self.past_grid_history + self.current_grid_history
        X, y = [], []
        for i in range(N, len(combined_grid_history[:-1])):
            X.append(np.array([keras.utils.to_categorical(frame, num_classes=4) for frame in combined_grid_history[i - N:i]]))
            y.append(np.array(keras.utils.to_categorical(combined_grid_history[i], num_classes=4)))
        return np.array(X), np.array(y)

    def augment_data(self, X, y):
        augmented_X, augmented_y = self.basic_augmentation(X, y)
        modified_X, modified_y = self.advanced_augmentation(augmented_X, augmented_y)
        return augmented_X, augmented_y, modified_X, modified_y

    def basic_augmentation(self, X, y):
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            # Original data
            augmented_X.append(X[i])
            augmented_y.append(y[i])

            # Horizontal flip
            X_hor_flip = np.flip(X[i], axis=1)
            y_hor_flip = np.flip(y[i], axis=0)
            augmented_X.append(X_hor_flip)
            augmented_y.append(y_hor_flip)

            # Vertical flip
            X_ver_flip = np.flip(X[i], axis=2)
            y_ver_flip = np.flip(y[i], axis=1)
            augmented_X.append(X_ver_flip)
            augmented_y.append(y_ver_flip)

            # Rotations 90, 180, 270 degrees
            for angle in [90, 180, 270]:
                X_rotated = scipy.ndimage.rotate(X[i], angle, axes=(1, 2), reshape=False)
                y_rotated = scipy.ndimage.rotate(y[i], angle, axes=(0, 1), reshape=False)
                augmented_X.append(X_rotated)
                augmented_y.append(y_rotated)
                
            # Translations
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                X_translated = np.roll(X[i], (dx, dy), axis=(1, 2))
                y_translated = np.roll(y[i], (dx, dy), axis=(0, 1))
                augmented_X.append(X_translated)
                augmented_y.append(y_translated)

        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)
        return augmented_X, augmented_y

    def advanced_augmentation(self, augmented_X, augmented_y, modification_rate=0.01, noise_level=0.01, time_distortion_weights=(0.7, 0.2, 0.1), warping_strength=0.1, num_agents_to_move=1):
        modified_X = []
        modified_y = []

        for i in range(len(augmented_X)):
            # Cell Type Modification Augmentation
            X_cell_mod = np.copy(augmented_X[i])
            num_modifications = int(modification_rate * X_cell_mod.size)
            indices = np.random.choice(X_cell_mod.size, num_modifications, replace=False)
            new_values = np.random.choice(np.array([0, 1, 2, 3]), num_modifications)
            np.put(X_cell_mod, indices, new_values)
            modified_X.append(X_cell_mod)
            modified_y.append(augmented_y[i])

            # Jittering Augmentation
            noise = noise_level * np.random.randn(*augmented_X[i].shape)
            X_noise_inject = augmented_X[i] + noise
            X_noise_inject = np.clip(X_noise_inject, 0, 3)  # Ensure the noisy data is within valid range
            modified_X.append(X_noise_inject)
            modified_y.append(augmented_y[i])

            # Time-Distortion Augmentation
            X_time_distort = np.copy(augmented_X[i])
            num_weights = len(time_distortion_weights)
            for t in range(num_weights - 1, augmented_X[i].shape[0]):
                X_time_distort[t] = sum(time_distortion_weights[j] * X_time_distort[t - j] for j in range(num_weights))
            modified_X.append(X_time_distort)
            modified_y.append(augmented_y[i])
            
            # Time Warping Augmentation
            num_knots = max(int(warping_strength * augmented_X[i].shape[1]), 2)
            seq_length = augmented_X.shape[1]
            original_indices = np.linspace(0, seq_length - 1, seq_length)
            knot_positions = np.linspace(0, seq_length - 1, num=num_knots, dtype=int)
            knot_offsets = warping_strength * np.random.randn(num_knots)
            warp_indices = np.clip(knot_positions + knot_offsets, 0, seq_length - 1)
            warp_function = interp1d(knot_positions, warp_indices, kind='linear', bounds_error=False)
            new_indices = warp_function(original_indices)
            new_indices = np.clip(new_indices, 0, seq_length - 1)
            signal = augmented_X[i]
            interp_function = interp1d(original_indices, signal, axis=0, kind='linear', bounds_error=False)
            warped_signal = interp_function(new_indices)
            warped_signal = np.clip(warped_signal, 0, 3)

            modified_X.append(warped_signal)
            modified_y.append(augmented_y[i])

            # Move a specified number of agents augmentation
            X_moved = np.copy(augmented_X[i])
            y_moved = np.copy(augmented_y[i])
            width, height = X_moved.shape[1], X_moved.shape[2]

            # Identify agent positions in both X and y
            agent_positions = np.argwhere(np.any(X_moved > 0, axis=-1) & np.any(y_moved > 0, axis=-1))
            np.random.shuffle(agent_positions)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                moved_agents = 0

                for t, x, y in agent_positions:
                    if moved_agents >= num_agents_to_move:
                        break

                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < width and 0 <= new_y < height:
                        # Check the validity of the new position
                        if not np.any(X_moved[:, max(0, new_x - 1):min(width, new_x + 2), max(0, new_y - 1):min(height, new_y + 2), :]) and not np.any(y_moved[new_x, new_y, :]):
                            # Move the agent across all timesteps
                            X_moved[:, x, y, :], X_moved[:, new_x, new_y, :] = X_moved[:, new_x, new_y, :], X_moved[:, x, y, :]
                            y_moved[x, y, :], y_moved[new_x, new_y, :] = y_moved[new_x, new_y, :], y_moved[x, y, :]
                            moved_agents += 1

            modified_X.append(X_moved)
            modified_y.append(y_moved)

        return np.array(modified_X), np.array(modified_y)

    def bootstrap_samples(self, basic_X, basic_y, advanced_X, advanced_y, n_basic_samples, n_advanced_samples):
        # Function to compute a representation score for each sample
        def compute_representation_score(y):
            # Assuming y is one-hot encoded, shape: (samples, width, height, classes)
            # Sum over width and height dimensions for each class
            state_presence = np.sum(y, axis=(1, 2))
            # Ignore state 0 (majority state) and sum the presence of all other states
            representation_score = np.sum(state_presence[:, 1:], axis=1)
            return representation_score

        # Compute representation scores for basic and advanced data
        basic_y_scores = compute_representation_score(basic_y)
        advanced_y_scores = compute_representation_score(advanced_y)

        # Convert scores to categorical strata (e.g., 'low', 'medium', 'high')
        # Adjust the quantile thresholds as per your specific data distribution
        basic_y_strata = pd.qcut(pd.Series(basic_y_scores), q=[0, .33, .66, 1], labels=False, duplicates='drop').fillna(0)
        advanced_y_strata = pd.qcut(pd.Series(advanced_y_scores), q=[0, .33, .66, 1], labels=False, duplicates='drop').fillna(0)

        # Stratified resampling
        bootstrapped_basic_X, bootstrapped_basic_y = resample(basic_X, basic_y, n_samples=n_basic_samples, replace=True, stratify=basic_y_strata)
        bootstrapped_advanced_X, bootstrapped_advanced_y = resample(advanced_X, advanced_y, n_samples=n_advanced_samples, replace=True, stratify=advanced_y_strata)
        
        bootstrapped_X = np.concatenate((bootstrapped_basic_X, bootstrapped_advanced_X), axis=0)
        bootstrapped_y = np.concatenate((bootstrapped_basic_y, bootstrapped_advanced_y), axis=0)
        
        return bootstrapped_X, bootstrapped_y

    def channel_wise_dropout(self, rate):
        def dropout_func(inputs, training=None):
            if training:
                # Get the input shape
                input_shape = inputs.shape
                noise_shape = (input_shape[0], 1, 1, input_shape[-1])

                # Apply dropout
                dropout_mask = tf.nn.dropout(tf.ones(noise_shape), rate=rate)
                return inputs * dropout_mask
            else:
                return inputs

        return keras.layers.Lambda(lambda x: dropout_func(x))

    @keras.saving.register_keras_serializable()
    class ChannelWiseDropout(keras.layers.Layer):
        def __init__(self, rate, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate

        @tf.function
        def call(self, inputs, training=None):
            if not training:
                return inputs

            # Get noise shape
            input_shape = tf.shape(inputs)
            batch_size = tf.slice(input_shape, [0], [1])
            channels = tf.slice(input_shape, [tf.subtract(tf.size(input_shape), 1)], [1])
            noise_shape = tf.concat(
                [batch_size, tf.ones(tf.subtract(tf.rank(inputs), 2), dtype=tf.int32), channels], axis=0)

            # Apply dropout
            dropout_mask = tf.nn.dropout(tf.ones(noise_shape), rate=self.rate)
            dropout_mask = tf.broadcast_to(dropout_mask, input_shape)
            return inputs * dropout_mask

        def get_config(self):
            config = super().get_config()
            config.update({"rate": self.rate})
            return config

    def create_model(self, input_shape, filters, kernel_size, dropout_rate, l2_regularizer, use_attention=True):
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # First ConvLSTM2D layer
        x = layers.LayerNormalization()(inputs)
        x = layers.GaussianDropout(dropout_rate)(x)
        x = self.ChannelWiseDropout(dropout_rate)(x)
        convlstm1 = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation='gelu', padding='same', return_sequences=True,
                                    recurrent_dropout=dropout_rate, recurrent_regularizer=keras.regularizers.l2(l2_regularizer), kernel_regularizer=keras.regularizers.l2(l2_regularizer), bias_regularizer=keras.regularizers.l2(l2_regularizer))(x)
        
        # LayerNormalization and Dropout after the first ConvLSTM2D
        x = layers.LayerNormalization()(convlstm1)
        x = layers.GaussianDropout(dropout_rate)(x)
        x = self.ChannelWiseDropout(dropout_rate)(x)
        
        # Second ConvLSTM2D layer
        convlstm2 = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation='gelu', padding='same', return_sequences=True,
                                    recurrent_dropout=dropout_rate, recurrent_regularizer=keras.regularizers.l2(l2_regularizer), kernel_regularizer=keras.regularizers.l2(l2_regularizer), bias_regularizer=keras.regularizers.l2(l2_regularizer))(x)

        # Adding residual connection
        x = layers.Add()([convlstm1, convlstm2])
        
        # LayerNormalization and Dropout after combining with the residual connection
        x = layers.LayerNormalization()(x)
        x = layers.GaussianDropout(dropout_rate)(x)
        x = self.ChannelWiseDropout(dropout_rate)(x)
        
        if use_attention:
            # Attention Mechanism
            reshaped_x = layers.Reshape((-1, input_shape[1]*input_shape[2]*filters))(x)
            attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=filters//16, dropout=dropout_rate,
                                        kernel_regularizer=keras.regularizers.l2(l2_regularizer), bias_regularizer=keras.regularizers.l2(l2_regularizer))(reshaped_x, reshaped_x)
            reshaped_attention_output = layers.Reshape((-1, input_shape[1], input_shape[2], filters))(attention_output)
            
            # Adding residual connection
            x = layers.Multiply()([x, reshaped_attention_output])
            
            # LayerNormalization and Dropout after combining with the residual connection
            x = layers.LayerNormalization()(x)
            x = layers.GaussianDropout(dropout_rate)(x)
            x = self.ChannelWiseDropout(dropout_rate)(x)
        
        # Reducing over the time dimension using GlobalAveragePooling
        x = layers.Reshape((-1, input_shape[1]*input_shape[2]*filters))(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Reshape((input_shape[1], input_shape[2], filters))(x)

        # Output layer
        outputs = layers.Conv2D(filters=4, kernel_size=(1, 1), activation='softmax')(x)
        
        # Create and compile the model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Nadam()
        
        custom_loss = self.combined_loss_function
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.TopKCategoricalAccuracy(k=2)])
        
        return model

    def compute_class_weights(self, y):
        y_indices = np.argmax(y, axis=-1) # Assuming y is one-hot encoded, convert to class indices
        class_weights = np.zeros((4,))
        unique_classes = np.unique(y_indices)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_indices.flatten())
        class_weights[unique_classes] = weights

        return class_weights

    @staticmethod
    @keras.utils.register_keras_serializable()
    def combined_loss_function(y_true, y_pred, alpha=0.25, gamma=2.0, label_smoothing=0.1, focal_loss_weight=0.25, weighted_loss_weight=0.25, crossentropy_loss_weight=0.25, count_loss_weight=0.25):
        weights = PredictionObserver.class_weights
        focal_loss_fn = keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=gamma)
        crossentropy_loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        def weighted_categorical_crossentropy(y_true, y_pred):
            y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
            loss = y_true * tf.math.log(y_pred) * weights
            loss = -tf.reduce_sum(loss, -1)
            return loss

        def count_loss(y_true, y_pred):
            true_count = tf.reduce_sum(y_true[..., 1], axis=[1, 2])
            pred_count = tf.reduce_sum(y_pred[..., 1], axis=[1, 2])
            return tf.reduce_mean(tf.square(true_count - pred_count))

        focal_loss = focal_loss_fn(y_true, y_pred)
        crossentropy_loss = crossentropy_loss_fn(y_true, y_pred)
        weighted_loss = weighted_categorical_crossentropy(y_true, y_pred)
        agent_count_loss = count_loss(y_true, y_pred)

        combined_loss = (tf.math.multiply(focal_loss_weight, focal_loss) +
                        tf.math.multiply(crossentropy_loss_weight, crossentropy_loss) +
                        tf.math.multiply(weighted_loss_weight, weighted_loss) +
                        tf.math.multiply(count_loss_weight, agent_count_loss))
        return combined_loss

    def cosine_annealing_scheduler(self, max_update=20, base_lr=0.01, final_lr=0.001, warmup_steps=5, warmup_begin_lr=0.001, cycle_length=10, exp_decay_rate=0.5):
        # Pre-compute constants for efficiency
        warmup_slope = (base_lr - warmup_begin_lr) / warmup_steps
        max_steps = max_update - warmup_steps

        def schedule(epoch):
            if epoch < warmup_steps:
                # Warmup phase with a linear increase
                return warmup_begin_lr + warmup_slope * epoch
            elif epoch < max_update:
                # Main learning phase with cosine annealing
                return final_lr + (base_lr - final_lr) * (1 + math.cos(math.pi * (epoch - warmup_steps) / max_steps)) / 2
            else:
                # Post-max_update phase with warm restarts and cosine annealing
                adjusted_epoch = epoch - max_update
                cycles = math.floor(1 + (adjusted_epoch - 1) / cycle_length)
                x = adjusted_epoch - (cycles * cycle_length)

                # Apply exponential decay to base_lr only when a new cycle begins
                decayed_lr = base_lr * (exp_decay_rate ** cycles)

                # Apply cosine annealing within the cycle
                cycle_base_lr = max(decayed_lr, final_lr)
                lr = final_lr + (cycle_base_lr - final_lr) * (1 - math.cos(math.pi * x / cycle_length)) / 2
                return max(lr, final_lr)  # Ensure lr does not go below final_lr

        return schedule

    def tune_hyperparameters(self, train_dataset, batch_size, num_folds, param_grid):
        best_loss = float('inf')
        best_params = {}
        
        dataset_size = len(list(train_dataset.as_numpy_iterator()))
        fold_size = dataset_size // num_folds

        for params in ParameterGrid(param_grid):
            fold_loss_scores = []

            for fold in range(num_folds):
                # Calculate start and end indices for the current fold
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold != num_folds - 1 else dataset_size

                # Split the dataset into training and validation sets for the current fold
                train_data_fold = train_dataset.skip(end_idx).take(dataset_size - end_idx).concatenate(train_dataset.take(start_idx))
                val_data_fold = train_dataset.skip(start_idx).take(fold_size)

                # Prepare the data folds
                train_data_fold = train_data_fold.prefetch(tf.data.experimental.AUTOTUNE)
                val_data_fold = val_data_fold.prefetch(tf.data.experimental.AUTOTUNE)

                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)
                lr_scheduler = keras.callbacks.LearningRateScheduler(self.cosine_annealing_scheduler())
                model = self.create_model(train_data_fold.element_spec[0].shape[1:], **params)
                history = model.fit(train_data_fold, epochs=20, validation_data=val_data_fold, verbose=0, callbacks=[early_stopping, lr_scheduler])

                loss = history.history['val_loss'][-1]
                fold_loss_scores.append(loss)

            avg_loss = np.mean(fold_loss_scores).item()
            
            print(f"Params: {params}, Avg Loss: {avg_loss}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params

        return best_params, best_loss

    def log_image_summary(self, epoch, model):
        def plot_to_image(figure):
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue())
            image = tf.expand_dims(image, 0)
            return image

        def create_comparison_figure(original, actual, predicted, uncertainty):
            # Convert from categorical to scalar values
            original = np.argmax(original, axis=-1)
            actual = actual.astype(int)
            predicted = np.argmax(predicted, axis=-1)
            uncertainty = np.max(uncertainty, axis=-1)
            
            num_original_images = original.shape[0]
            figure = plt.figure(figsize=(2 * num_original_images, 4), constrained_layout=True)

            # Plotting original states
            for i in range(num_original_images):
                ax = figure.add_subplot(2, num_original_images + 3, i + 1)
                ax.imshow(original[i], cmap='viridis')
                ax.axis('off')
                ax.set_title("Original" if i == 0 else "")

            # Plotting actual state in the second row, first position
            ax = figure.add_subplot(2, num_original_images + 3, num_original_images + 4)
            ax.imshow(actual, cmap='viridis')
            ax.axis('off')
            ax.set_title("Actual")

            # Plotting predicted state in the second row, second position
            ax = figure.add_subplot(2, num_original_images + 3, num_original_images + 5)
            ax.imshow(predicted[0], cmap='viridis')
            ax.axis('off')
            ax.set_title("Predicted")

            # Plotting uncertainty in the second row, third position
            ax = figure.add_subplot(2, num_original_images + 3, num_original_images + 6)
            ax.imshow(uncertainty, cmap='hot', interpolation='nearest')
            ax.axis('off')
            ax.set_title("Uncertainty")
                
            return figure

        original = np.array([keras.utils.to_categorical(frame, num_classes=4) for frame in self.current_grid_history[-6:-1]])
        actual_state = self.current_grid_history[-1]
        predicted = model.predict(original.reshape((1, -1, self.subject.school.size, self.subject.school.size, 4)), verbose=0)
        uncertainty = self.monte_carlo_prediction(model, original.reshape((1, -1, self.subject.school.size, self.subject.school.size, 4)), n_predictions=100)

        figure = create_comparison_figure(original, actual_state, predicted, uncertainty)
        image = plot_to_image(figure)

        # Log the image to TensorBoard
        with self.file_writer.as_default():
            tf.summary.image("Comparison: Original, Actual, Predicted, Uncertainty", image, step=epoch)

    def train_model(self, model, num_steps=5, num_folds=5, batch_size=16, n_basic_samples=100, n_advanced_samples=50):
        # X should have shape (samples, timesteps, width, height, channels)
        # y should have shape (samples, width, height, channels)
        X, y = self.prepare_data(num_steps)
        augmented_X, augmented_y, modified_X, modified_y = self.augment_data(X, y)
        X_boot, y_boot = self.bootstrap_samples(augmented_X, augmented_y, modified_X, modified_y, n_basic_samples, n_advanced_samples)
        X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=0.2, random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=min(len(X_train), 1024)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        class_weights = self.compute_class_weights(y_train)
        PredictionObserver.class_weights = tf.Variable(class_weights, dtype=tf.float32)

        if model is None:
            param_grid = {
                'filters': [16],
                'kernel_size': [(3, 3)],
                'dropout_rate': [0.1, 0.3],
                'l2_regularizer': [0.0001]
            }
            best_params, best_loss = self.tune_hyperparameters(train_dataset, batch_size, num_folds, param_grid)
            print(f"Best Params: {best_params}, Best Loss: {best_loss}")
            model = self.create_model(train_dataset.element_spec[0].shape[1:], **best_params)
        
        model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        lr_scheduler = keras.callbacks.LearningRateScheduler(self.cosine_annealing_scheduler())
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.data_dir, "best_model.keras"), monitor="val_categorical_accuracy", mode='max', verbose=0, save_best_only=True)
        tensorboard = keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_graph=False)
        image_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.log_image_summary(epoch, model))
        model.fit(train_dataset, epochs=20, verbose=0, callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard, image_callback], validation_data=test_dataset)
        test_loss, test_f1_score = self.evaluate_model(model, X_test, y_test)
        print(f"Loss on the test set: {test_loss}")
        print(f"F1 Score on the test set: {test_f1_score}")
        return model

    def evaluate_model(self, model, X_test, y_test):
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        
        y_pred = model.predict(X_test, verbose=0)
        y_test_flat = y_test.argmax(axis=-1).flatten()
        y_pred_flat = y_pred.argmax(axis=-1).flatten()
        test_f1_score = f1_score(y_test_flat, y_pred_flat, average='micro')

        return test_loss, test_f1_score

    def monte_carlo_prediction(self, model, input_data, n_predictions=100):
        batched_input = np.repeat(np.array(input_data), n_predictions, axis=0)
        model._training = True
        batched_predictions = model.predict_on_batch(batched_input)
        model._training = False
        prediction_std = np.std(batched_predictions, axis=0)
        return prediction_std

    def plot_combined_heatmaps(self, past_grid_state, actual, predicted, uncertainty):
        fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

        # Plot the input timesteps
        for i in range(5):
            sns.heatmap(past_grid_state[i], ax=axs[0, i], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
            axs[0, i].set_title(f"Input Time {i+1}")
            axs[0, i].axis('off')

        # Plot the actual, predicted, and uncertainty heatmaps
        sns.heatmap(actual, ax=axs[1, 0], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
        axs[1, 0].set_title("Actual State")
        axs[1, 0].axis('off')

        sns.heatmap(predicted, ax=axs[1, 1], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
        axs[1, 1].set_title("Predicted State")
        axs[1, 1].axis('off')

        sns.heatmap(uncertainty, ax=axs[1, 2], cmap='hot', cbar=True, square=True)
        axs[1, 2].set_title("Uncertainty")
        axs[1, 2].axis('off')

        for i in range(3, 5):
            axs[1, i].axis('off')  # Hide unused subplots

        plt.show()

    def calculate_ssim(self, actual, predicted):
        # Flatten the grid states to 2D arrays for SSIM calculation
        actual_flat = actual.astype(float).reshape((self.subject.school.size, self.subject.school.size))
        predicted_flat = predicted.astype(float).reshape((self.subject.school.size, self.subject.school.size))
        return ssim(actual_flat, predicted_flat, data_range=3)

    def calculate_nrmse(self, actual, predicted):
        # Flatten the grid states to 2D arrays for NRMSE calculation
        actual_flat = actual.reshape((self.subject.school.size, self.subject.school.size))
        predicted_flat = predicted.reshape((self.subject.school.size, self.subject.school.size))
        # Calculate RMSE
        mse = np.mean((actual_flat - predicted_flat) ** 2)
        rmse = np.sqrt(mse)
        # Normalize RMSE
        range_actual = actual_flat.max() - actual_flat.min()
        nrmse = rmse / range_actual if range_actual != 0 else float('inf')
        
        return nrmse

    def save_simulation_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        # Only save current grid history if it has multiple states
        if len(self.current_grid_history) > 1 and any(
            not np.array_equal(self.current_grid_history[i], self.current_grid_history[i + 1])
            for i in range(len(self.current_grid_history) - 1)
        ):
            self.past_grid_history.extend(self.current_grid_history)
            np.savez(self.grid_history_path, grid_history=self.past_grid_history)
            print(f"Saved {len(self.past_grid_history)} states to {self.grid_history_path}")
        else:
            print("No new states to save. There are {self.past_grid_history} states in the history.")

    def run_tensorboard(self):
        tb = program.TensorBoard(plugins=default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.log_dir])
        tb.launch()
        tensorboard_url = 'http://localhost:6006/'
        print(f"TensorBoard is running at {tensorboard_url}")

        # Start Chrome with Selenium
        chrome_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=chrome_service)
        driver.get(tensorboard_url)
        print(f"Chrome started at {tensorboard_url}")

        return driver

    def display_observation(self, train_model=True):
        if os.path.exists(os.path.join(self.data_dir, "best_model.keras")):
            self.model = keras.models.load_model(os.path.join(self.data_dir, "best_model.keras"), custom_objects={'combined_loss_function': PredictionObserver.combined_loss_function, 'ChannelWiseDropout': PredictionObserver.ChannelWiseDropout})
            print("Loaded the best model from previous training.")
        else:
            self.model = getattr(self, "model", None)
            print("No saved model found.")
        if train_model:
            self.model = self.train_model(self.model)
            print(f"Trained the model with {len(self.current_grid_history)+len(self.past_grid_history)-1} states.")
        elif self.model is not None:
            print("Model already exists. Skipping training.")
        else:
            raise ValueError("Model is None. Please train the model first.")

        if self.model is None:
            raise ValueError("Model is None. Please train the model first.")
        elif not isinstance(self.model, keras.models.Model) or not self.model.built:
            raise ValueError("Model is not properly instantiated.")

        self.save_simulation_data()

        past_grid_state = self.current_grid_history[-6:-1]
        argmax_past_grid_state = keras.utils.to_categorical(past_grid_state, num_classes=4)
        input_data = np.array(argmax_past_grid_state).reshape((1, -1, self.subject.school.size, self.subject.school.size, 4))
        predicted_grid_state = self.model.predict(input_data, verbose=0)
        argmax_predicted_grid_state = np.argmax(predicted_grid_state, axis=-1)
        reshaped_grid_state = argmax_predicted_grid_state.reshape((self.subject.school.size, self.subject.school.size))
        
        prediction_std = self.monte_carlo_prediction(self.model, input_data, n_predictions=100)
        state_uncertainty = np.max(prediction_std, axis=-1).reshape((self.subject.school.size, self.subject.school.size))
        
        self.plot_combined_heatmaps(past_grid_state, self.current_grid_history[-1].astype(int), reshaped_grid_state, state_uncertainty)
        
        print("Actual State:")
        print(self.current_grid_history[-1].astype(int))
        print("Predicted State:")
        reformatted_grid_state = np.round(reshaped_grid_state, 1)
        print(reformatted_grid_state)
        print("Uncertainty (Std in Predicted Probabilities):")
        reformatted_state_uncertainty = np.array2string(state_uncertainty, formatter={'float_kind':lambda x: "{:.0e}".format(x)})
        print(reformatted_state_uncertainty)
        
        ssim_value = self.calculate_ssim(self.current_grid_history[-1].astype(int), reshaped_grid_state)
        print(f"SSIM: {ssim_value:.3f}")
        nrmse_value = self.calculate_nrmse(self.current_grid_history[-1].astype(int), reshaped_grid_state)
        print(f"NRMSE: {nrmse_value:.3f}")

        if train_model:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.run_tensorboard)
                driver = future.result()

                builtins.input("Press Enter to stop TensorBoard and close the Chrome window.")

                # Close Chrome
                driver.quit()
            # After pressing Enter, the script will continue from here
            print("Stopping TensorBoard and exiting the script.")


class FFTAnalysisObserver(Observer):
    def __init__(self, population):
        self.subject = population
        self.subject.attach_observer(self)
        self.spatial_data = []
        self.time_series_data = []

    def update(self):
        if self.subject.timestep == 1:
            print("Initializing FFT Analysis")
            self.time_series_data.append(0)
        self.spatial_data.append(self.capture_grid_state())
        self.time_series_data.append(self.count_zombies())

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for individual in self.subject.agent_list:
            grid_state[individual.location] = individual.health_state.value if individual else 0
        return grid_state

    def count_zombies(self):
        return sum(ind.health_state == HealthState.ZOMBIE for ind in self.subject.agent_list)

    def perform_fft_analysis(self):
        self.spatial_fft = [fftshift(fft2(frame)) for frame in self.spatial_data]
        self.time_series_fft = fft(self.time_series_data)
        magnitudes = np.abs(self.time_series_fft)
        self.frequencies = np.fft.fftfreq(len(self.time_series_data), d=1)
        self.dominant_frequencies = self.frequencies[np.argsort(-magnitudes)[:5]]
        self.dominant_periods = [1 / freq if freq != 0 else float('inf') for freq in self.dominant_frequencies]

    def create_spatial_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("Spatial Data Over Time")
        im = ax.imshow(self.spatial_data[0], cmap='viridis', animated=True)
        plt.colorbar(im, ax=ax, orientation='vertical')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")

        def update(frame):
            im.set_array(self.spatial_data[frame])
            time_text.set_text(f"Time Step: {frame}")
            return [im, time_text]

        ani = animation.FuncAnimation(fig, update, frames=len(self.spatial_data), interval=50, blit=True)
        plt.show()

    def create_spatial_fft_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("FFT of Spatial Data Over Time")
        fft_data = np.log(np.abs(self.spatial_fft[0]) + 1e-10)
        im = ax.imshow(fft_data, cmap='hot', animated=True)
        plt.colorbar(im, ax=ax, orientation='vertical')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")

        def update(frame):
            fft_data = np.log(np.abs(self.spatial_fft[frame]) + 1e-10)
            im.set_array(fft_data)
            time_text.set_text(f"Time Step: {frame}")
            return [im, time_text]

        ani = animation.FuncAnimation(fig, update, frames=len(self.spatial_data), interval=50, blit=True)
        plt.show()

    def create_time_series_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("Time Series Data")
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(-1, len(self.time_series_data) + 1)
        ax.set_ylim(-1, max(self.time_series_data) + 1)

        # Mark periods and dominant frequencies on the plot
        plotted_periods = set()
        for period in self.dominant_periods:
            if period != float('inf') and period not in plotted_periods and period > 0:
                ax.axvline(x=period, color='r', linestyle='--', label=f'Period: {period:.2f} steps')
                plotted_periods.add(period)

        def update(frame):
            line.set_data(np.arange(frame), self.time_series_data[:frame])
            return line,

        ani = animation.FuncAnimation(fig, func=update, frames=len(self.time_series_data)+1, interval=50, blit=True)
        ax.legend(loc='upper right')
        plt.show()

    def create_time_series_fft_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("FFT of Time Series Data")
        line, = ax.plot([], [], lw=2, label='FFT')
        ax.set_xlim(min(self.frequencies), max(self.frequencies))
        ax.set_ylim(-1, max(np.abs(self.time_series_fft)) + 1)

        # Mark dominant frequencies on the plot
        plotted_frequencies = set()
        for freq in self.dominant_frequencies:
            if freq not in plotted_frequencies:
                ax.axvline(x=freq, color='r', linestyle='--', label=f'Frequency: {freq:.2f}')
                plotted_frequencies.add(freq)

        def update(frame):
            frame_data = self.time_series_data[:frame + 1]
            fft_frame = fft(frame_data)
            freqs = np.fft.fftfreq(len(frame_data), d=1)
            line.set_data(freqs, np.abs(fft_frame))
            return line,

        ani = animation.FuncAnimation(fig, func=update, frames=len(self.time_series_data)+1, interval=50, blit=True)
        ax.legend(loc='upper right')
        plt.show()

    def plot_final_spatial_data(self, ax):
        ax.imshow(self.spatial_data[-1], cmap='viridis')
        ax.set_title("Final Spatial Data")
        ax.figure.colorbar(ax.images[0], ax=ax, orientation='vertical')

    def plot_fft_final_spatial_data(self, ax):
        ax.imshow(np.log(np.abs(self.spatial_fft[-1]) + 1e-10), cmap='hot')
        ax.set_title("FFT of Final Spatial Data")
        ax.figure.colorbar(ax.images[0], ax=ax, orientation='vertical')

    def plot_time_series_data(self, ax):
        ax.plot(self.time_series_data)
        for period in self.dominant_periods:
            if period != float('inf') and period > 0:
                ax.axvline(x=period, color='r', linestyle='--', label=f'Period: {period:.2f} steps')
        ax.set_title("Time Series Data")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Number of Zombies")
        ax.legend()

    def plot_fft_time_series_data(self, ax):
        ax.plot(self.frequencies, np.abs(self.time_series_fft))
        for freq in self.dominant_frequencies:
            ax.axvline(x=freq, color='r', linestyle='--', label=f'Frequency: {freq:.2f}')
        ax.set_title("FFT of Time Series Data")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    def display_observation(self, mode='static'):
        if not self.spatial_data or not self.time_series_data:
            print("No data available for FFT analysis.")
            return

        self.perform_fft_analysis()

        if mode == 'animation':
            self.create_spatial_animation()
            self.create_spatial_fft_animation()
            self.create_time_series_animation()
            self.create_time_series_fft_animation()
        elif mode == 'static':
            fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)

            self.plot_final_spatial_data(axs[0, 0])
            self.plot_fft_final_spatial_data(axs[0, 1])
            self.plot_time_series_data(axs[1, 0])
            self.plot_fft_time_series_data(axs[1, 1])

            plt.show()

class PygameObserver(Observer):
    def __init__(self, population, cell_size=30, fps=10, font_size=18):
        self.subject = population
        self.subject.attach_observer(self)

        # Define the cell size, the frames per second, and font size
        self.cell_size = cell_size
        self.fps = fps
        self.font_size = font_size
        self.is_paused = False

        # Colors
        self.colors = {
            HealthState.HEALTHY: (0, 255, 0),  # Green
            HealthState.INFECTED: (255, 165, 0),  # Orange
            HealthState.ZOMBIE: (255, 0, 0),  # Red
            HealthState.DEAD: (128, 128, 128),  # Gray
            'background': (255, 255, 255),  # White
            'grid_line': (200, 200, 200),  # Light Gray
            'text': (0, 0, 0)  # Black
        }

        # Initialize Pygame and the screen
        pygame.init()
        self.screen_size = self.subject.school.size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + 50))  # Additional space for stats
        pygame.display.set_caption("Zombie Apocalypse Simulation")

        # Clock for controlling the frame rate
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", self.font_size)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.handle_quit_event()
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown_events(event)

    def handle_quit_event(self):
        pygame.quit()
        exit()

    def handle_keydown_events(self, event):
        if event.key == pygame.K_SPACE:
            self.toggle_pause()
        elif event.key == pygame.K_r:
            pass
            # Restart the simulation

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.display_pause_message()

    def update(self):
        if self.is_paused:
            while self.is_paused:
                self.handle_events()  # Continue to handle events while paused to catch unpause event
                pygame.time.wait(10)  # Wait for a short period to reduce CPU usage while paused
        else:
            self.handle_events()  # Handle events for unpausing or quitting
            self.draw_grid()
            self.display_stats()
            pygame.display.flip()
            self.clock.tick(self.fps)
            time.sleep(0.5)

    def draw_grid(self):
        # If not paused, fill the background and draw individuals
        if not self.is_paused:
            self.screen.fill(self.colors['background'])
            self.draw_individuals()

    def draw_individuals(self):
        for individual in self.subject.agent_list:
            x, y = individual.location
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[individual.health_state], rect)
            pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 1)  # Draw grid line

    def display_stats(self):
        stats_text = f"Healthy: {self.subject.num_healthy}, Infected: {self.subject.num_infected}, Zombie: {self.subject.num_zombie}, Dead: {self.subject.num_dead}"
        text_surface = self.font.render(stats_text, True, self.colors['text'])
        self.screen.fill(self.colors['background'], (0, self.screen_size, self.screen_size, 50))  # Clear stats area
        self.screen.blit(text_surface, (5, self.screen_size + 5))

    def display_pause_message(self):
        dark_surface = pygame.Surface((self.screen_size, self.screen_size))
        dark_surface.set_alpha(128)
        dark_surface.fill(self.colors['background'])
        self.screen.blit(dark_surface, (0, 0))
        pause_text = "Simulation Paused. Press 'Space' to resume."
        text_surface = self.font.render(pause_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.screen_size / 2 - text_surface.get_width() / 2, self.screen_size / 2 - text_surface.get_height() / 2))

    def display_observation(self):
        end_text = "Simulation Ended. Press 'R' to Restart."
        text_surface = self.font.render(end_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.screen_size / 2 - text_surface.get_width() / 2, self.screen_size / 2 - text_surface.get_height() / 2))
        pygame.display.flip()
        while True:
            self.handle_events()

class GANObserver:
    def __init__(self, population, learning_rate=0.00005):
        self.subject = population
        self.subject.attach_observer(self)
        self.learning_rate = learning_rate
        self.data_shape = (self.subject.school.size, self.subject.school.size)
        self.latent_dim = self.subject.school.size
        self.num_classes = 4
        self.generator = self.build_generator()
        self.generator.summary()
        self.critic = self.build_critic()
        self.critic.summary()
        self.critic, self.generator, self.critic_optimizer, self.generator_optimizer = self.build_gan(self.critic, self.generator)
        self.real_data_samples = []
        self.timesteps = []

    def build_generator(self, num_layers=1, filter_size=16, dropout_rate=0.25):
        noise_input = layers.Input(shape=self.data_shape + (1,))
        timestep_input = layers.Input(shape=(1,))
        
        timestep_dense = layers.Dense(self.data_shape[0] * self.data_shape[1], use_bias=False)(timestep_input)
        timestep_reshaped = layers.Reshape(self.data_shape + (1,))(timestep_dense)
        merged_input = layers.Add()([noise_input, timestep_reshaped])
        
        x = layers.Flatten()(merged_input)
        x = layers.Dense((self.data_shape[0] - 2 * num_layers - 2) * (self.data_shape[1] - 2 * num_layers - 2) * filter_size)(x)
        x = layers.Reshape((self.data_shape[0] - 2 * num_layers - 2, self.data_shape[1] - 2 * num_layers - 2, filter_size))(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout_rate)(x)

        for _ in range(num_layers):
            x = layers.Conv2DTranspose(filter_size, kernel_size=(3, 3), padding='valid')(x)
            x = layers.ELU()(x)
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv2DTranspose(self.num_classes, kernel_size=(3, 3), padding='valid', activation="softmax")(x)

        model = keras.models.Model(inputs=[noise_input, timestep_input], outputs=x)
        return model

    def build_critic(self, num_layers=1, filter_size=32, dropout_rate=0.25, m=2):
        # pacgan implementation
        data_input = layers.Input(shape=self.data_shape + (self.num_classes * m,))
        timestep_input = layers.Input(shape=(m,))
        
        timestep_dense = layers.Dense(int(np.prod(self.data_shape)) * self.num_classes * m, use_bias=False)(timestep_input)
        timestep_reshaped = layers.Reshape(self.data_shape + (self.num_classes * m,))(timestep_dense)
        merged_input = layers.Add()([data_input, timestep_reshaped])
        
        x = layers.GaussianNoise(0.1)(merged_input)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout_rate)(x)

        for _ in range(num_layers):
            x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='valid', kernel_constraint=keras.constraints.MinMaxNorm(min_value=-0.01, max_value=0.01, rate=1.0))(x)
            x = layers.ELU()(x)
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='linear')(x)
        
        model = keras.models.Model(inputs=[data_input, timestep_input], outputs=x)
        return model

    @staticmethod
    def gradient_penalty(batch_size, real_images, fake_images, critic, labels, strength):
        alpha_shape = [batch_size, 1, 1, 1]
        alpha = tf.random.uniform(shape=alpha_shape, minval=0, maxval=1)
        
        interpolated_images = (real_images * alpha) + (fake_images * (1 - alpha))
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            prediction = critic([interpolated_images, labels], training=True)
        
        gradients = tape.gradient(prediction, [interpolated_images])
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2) * strength
        return penalty

    def build_gan(self, critic, generator):
        # Wasserstein loss function
        def wasserstein_loss(y_true, y_pred):
            return tf.reduce_mean(y_true * y_pred)

        critic_optimizer = keras.optimizers.Nadam(learning_rate=self.learning_rate)
        generator_optimizer = keras.optimizers.Nadam(learning_rate=self.learning_rate)

        critic.compile(loss=wasserstein_loss, optimizer=critic_optimizer)
        generator.compile(loss=wasserstein_loss, optimizer=generator_optimizer)
        
        return critic, generator, critic_optimizer, generator_optimizer

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                individual = self.subject.school.get_individual((i, j))
                grid_state[i, j] = individual.health_state.value if individual else 0
        return grid_state

    def update(self):
        real_data = self.capture_grid_state()
        self.real_data_samples.append(real_data)
        self.timesteps.append(np.array([self.subject.timestep]))

    @staticmethod
    def critic_training_step(critic, generator, data_shape, batch_size, num_classes, critic_optimizer, real_images, real_labels, gradient_penalty, m):
        noise = tf.random.normal([batch_size * m, data_shape[0], data_shape[1], 1])
        fake_labels = tf.random.uniform([batch_size * m, 1], 0, num_classes, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            fake_images = generator([noise, fake_labels], training=True)
            fake_images_packed = tf.reshape(fake_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
            
            selected_indices = tf.random.shuffle(tf.range(start=0, limit=batch_size * m, delta=1))
            shuffled_real_images = tf.gather(real_images, selected_indices, axis=0)
            real_images_packed = tf.reshape(shuffled_real_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
            shuffled_real_labels = tf.gather(real_labels, selected_indices, axis=0)
            real_labels_packed = tf.reshape(shuffled_real_labels, (batch_size, m))
            
            real_output = critic([real_images_packed, real_labels_packed], training=True)
            fake_output = critic([fake_images_packed, real_labels_packed], training=True)
            
            critic_real_loss = tf.reduce_mean(fake_output)
            critic_fake_loss = -tf.reduce_mean(real_output)
            gp = gradient_penalty(batch_size, real_images_packed, fake_images_packed, critic, real_labels_packed, strength=10.0)
            
            critic_loss = critic_real_loss + critic_fake_loss + gp
        
        critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables)) if critic_gradients else None
        
        return critic_loss, critic_real_loss, critic_fake_loss

    @staticmethod
    def generator_training_step(generator, critic, data_shape, batch_size, num_classes, generator_optimizer, m):
        noise = tf.random.normal([batch_size * m, data_shape[0], data_shape[1], 1])
        fake_labels = tf.random.uniform([batch_size * m, 1], 0, num_classes, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            generated_images = generator([noise, fake_labels], training=True)
            generated_images_packed = tf.reshape(generated_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
            packed_labels = tf.reshape(fake_labels, (batch_size, m))

            gen_output = critic([generated_images_packed, packed_labels], training=True)
            generator_loss = -tf.reduce_mean(gen_output)
        
        generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables)) if generator_gradients else None
        
        return generator_loss

    def train_gan(self, epochs=20, batch_size=128, critic_interval=2, generator_interval=1, m=2):
        for epoch in range(epochs):
            critic_losses, critic_real_losses, critic_fake_losses, generator_losses = [], [], [], []
            for _ in range(critic_interval):
                real_labels = tf.random.uniform([batch_size * m, 1], 0, self.num_classes, dtype=tf.int32)
                real_images = tf.one_hot(tf.random.uniform([batch_size * m, *self.data_shape], 0, self.num_classes, dtype=tf.int32), depth=self.num_classes)
                critic_loss, critic_real_loss, critic_fake_loss = self.critic_training_step(
                    self.critic, self.generator, self.data_shape, batch_size, self.num_classes, self.critic_optimizer, real_images, real_labels, self.gradient_penalty, m
                )
                critic_losses.append(critic_loss)
                critic_real_losses.append(critic_real_loss)
                critic_fake_losses.append(critic_fake_loss)
                
            for _ in range(generator_interval):
                generator_loss = self.generator_training_step(
                    self.generator, self.critic, self.data_shape, batch_size, self.num_classes, self.generator_optimizer, m
                )
                generator_losses.append(generator_loss)
            
            avg_critic_loss = np.mean([loss.numpy() for loss in critic_losses])
            avg_critic_real_loss = np.mean([loss.numpy() for loss in critic_real_losses])
            avg_critic_fake_loss = np.mean([loss.numpy() for loss in critic_fake_losses])
            avg_generator_loss = np.mean([loss.numpy() for loss in generator_losses])
            print(f"Epoch {epoch + 1}/{epochs} \t[ Critic Loss: {avg_critic_loss:.4f}, Critic Real Loss: {avg_critic_real_loss:.4f}, Critic Fake Loss: {avg_critic_fake_loss:.4f}, Generator Loss: {avg_generator_loss:.4f} ]")

    def display_observation(self):
        self.train_gan()
        noise = np.random.normal(0, 1, (1, self.data_shape[0], self.data_shape[1], 1))
        current_timestep = np.array([self.subject.timestep]).reshape((1, 1))
        generated_data = self.generator.predict([noise, current_timestep], verbose=0)
        generated_data = np.argmax(generated_data, axis=-1).reshape((self.subject.school.size, self.subject.school.size))
        print(generated_data)
