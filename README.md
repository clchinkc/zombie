
# Zombie Apocalypse Simulation in a School Environment

This project simulates a zombie apocalypse in a school setting, employing various Python classes and functions to model the interactions between humans, zombies, and the environment. The simulation provides insights into the spread of an infection in a confined space and includes visualization and analysis tools for in-depth exploration.

## Features

- **Simulation of a Zombie Apocalypse:** Models interactions between humans, zombies, and the environment within a school layout.
- **Object-Oriented Design:** Utilizes classes to represent individuals, zombies, and the school environment.
- **State Machine Pattern:** Manages state transitions (healthy, infected, zombie, dead) for individuals.
- **Movement Strategies:** Implements different movement strategies (random, flee, chase) based on the state of the individual.
- **Grid Management:** Handles legal movements, neighbor detection, and interactions within a 2D grid.
- **Statistical Tracking:** Tracks population metrics (counts of healthy, infected, zombies, dead) over time.
- **Visualization and Analysis:** Includes graphical tools (Matplotlib, Seaborn, Pygame, Tkinter) for visualizing the simulation progress and outcomes.
- **Machine Learning Analysis:** Applies predictive modeling to forecast future states of the simulation.


## Installation & Usage

Ensure you have Python 3.8 or newer installed for manual setup. For Docker, Docker must be installed on your system.

### Manual Installation

1. Clone the repository and install the required dependencies:

    ```bash
    git clone https://github.com/clchinkc/zombie.git
    pip install -r requirements.txt
    ```

2. Run the simulation:

    ```bash
    python main.py
    ```

### Installation Through TestPyPI

You may also install a stable release through TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps zombie_simulation
```

### Using Docker

To run the simulation using Docker, follow these steps:

1. Build the Docker Image:

    ```bash
    docker build -t zombie-simulation .
    ```

2. Run the Simulation in a Docker Container:

    ```bash
    docker run -it --rm zombie-simulation
    ```

    Note that the docker container will not display the graphical output. To view the graphical output, you can run the simulation locally.


## Components

- **Person Class:** Models an individual within the school, including attributes like location, state, and health.
- **Zombie Class:** Represents zombies with specific behaviors like attacking and infecting.
- **School Class:** Represents the school environment as a 2D grid and handles the dynamics within it.
- **Simulation Function:** Initializes and runs the simulation, updating and tracking the progress.
- **Visualization and Analysis Tools:** Provide graphical representations and analytical insights into the simulation's progression.

## Observers

The project includes several observers for analysis and visualization:

- **Simulation Observer:** Tracks the simulation's state and provides statistical output.
- **Animation Observers:** Generate animations showing the simulation's progress over time.
- **Tkinter Observer:** Displays the simulation in a graphical window using Tkinter.
- **Prediction Observer:** Uses machine learning to predict future states of the simulation.
- **FFT Analysis Observer:** Analyzes the frequency components of the simulation's dynamics.
- **Pygame Observer:** Visualizes the simulation in real-time using Pygame.
- **GAN Observer:** Generates realistic simulations of future states using Generative Adversarial Networks.

## Customization

The simulation parameters, such as the school size, population size, and number of time steps, can be adjusted in the `main.py` script. Additionally, the movement strategies and state transitions can be customized within their respective classes and methods.

## Contributing

Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
