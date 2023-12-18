import numpy as np


def set_up_cities():
    # Example setup, you can modify this as needed
    return [
        ["City1", False, [1, 2]],  # City 0
        ["City2", False, [0, 2]],  # City 1
        ["City3", False, [0, 1]]   # City 2
        # Add more cities as needed
    ]

def zombify(cities, cityno):
    cities[cityno][1] = True

def cure(cities, cityno):
    if cityno != 0:  # Ensuring city 0 always has zombies
        cities[cityno][1] = False

def sim_step(cities, p_spread, p_cure):
    for i, city in enumerate(cities):
        if city[1]:  # If the city is infected
            # Spread the infection
            if np.random.rand() < p_spread and city[2]:
                victim_city_no = np.random.choice(city[2])
                zombify(cities, victim_city_no)

            # Cure the infection
            if np.random.rand() < p_cure:
                cure(cities, i)

def draw_world(cities):
    for city in cities:
        print(city[0], "is", "infected" if city[1] else "healthy")

# Example usage
my_world = set_up_cities()
zombify(my_world, 0)  # Initially infect City 0

# Example of running a simulation step
sim_step(my_world, 0.5, 0.1)  # Adjust p_spread and p_cure as needed

# Example of drawing the world
draw_world(my_world)

