from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf
from observers import (
    FFTAnalysisObserver,
    GANObserver,
    MatplotlibAnimator,
    PlotlyAnimator,
    PredictionObserver,
    PygameObserver,
    SimulationAnimator,
    SimulationObserver,
    TkinterObserver,
)
from population import Population


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(0)

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=10)

    # create Observer objects
    # simulation_observer = SimulationObserver(school_sim)    # matplotlib not working in docker
    # simulation_animator = SimulationAnimator(school_sim)    # matplotlib not working in docker
    # plotly_animator = PlotlyAnimator(school_sim)    # plotly not working in docker
    # matplotlib_animator = MatplotlibAnimator(school_sim)    # matplotlib not working in docker
    # tkinter_observer = TkinterObserver(school_sim)  # tkinter not working in docker
    # prediction_observer = PredictionObserver(school_sim)    # selenium and matplotlib not working in docker
    # fft_observer = FFTAnalysisObserver(school_sim)    # matplotlib not working in docker
    # pygame_observer = PygameObserver(school_sim)    # pygame not working in docker
    gan_observer = GANObserver(school_sim)

    # run the population for a given time period
    school_sim.run_population(num_time_steps=100)
    
    print("Observers:")
    # print(simulation_observer.agent_list)
    # print(simulation_animator.agent_history[-1])

    # observe the statistics of the population
    # simulation_observer.display_observation(format="bar") # "statistics" or "grid" or "bar" or "scatter" or "table"
    # simulation_animator.display_observation(format="bar") # "bar" or "scatter" or "table"
    # plotly_animator.display_observation()
    # matplotlib_animator.display_observation()
    # tkinter_observer.display_observation()
    # prediction_observer.display_observation()
    # fft_observer.display_observation(mode='static') # "animation" or "static"
    # pygame_observer.display_observation()
    gan_observer.display_observation()


if __name__ == "__main__":
    main()