
"""
The Gravity Model is a mathematical algorithm used to predict the movement of people, goods, or information between two locations. It is based on the idea that the movement of entities is proportional to the mass of the entities and inversely proportional to the distance between them. 

The basic formula for the gravity model is:

Mij = Ci * Dj * f(Dij)

where:

- Mij: The predicted movement between location i and location j
- Ci: The population or economic activity at location i
- Dj: The population or economic activity at location j
- Dij: The distance between location i and location j
- f(Dij): A function of the distance between the two locations that typically follows an inverse power law, where the strength of the relationship between the two locations decreases as the distance between them increases.

The gravity model assumes that the movement between two locations is a function of both the attractiveness of the destination and the accessibility of the origin. This means that the movement will be higher between locations with higher populations or economic activity and lower distances between them.

The gravity model can be applied to a wide range of scenarios, such as predicting trade flows between countries, estimating migration patterns between cities, or modeling the diffusion of information between social networks. The model can be modified to include additional variables such as transportation costs, infrastructure, or cultural similarities to improve its accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

cities = [
    {'name': 'City A', 'population': 1_000_000, 'location': (0, 0), 'color': 'red'},
    {'name': 'City B', 'population': 2_000_000, 'location': (500, 500), 'color': 'blue'},
    {'name': 'City C', 'population': 1_500_000, 'location': (300, 800), 'color': 'green'},
    {'name': 'City D', 'population': 500_000, 'location': (1000, 100), 'color': 'orange'},
]

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def gravity_model(Ci, Dj, Dij, alpha=-2):
    return Ci * Dj * (Dij ** alpha)

n_cities = len(cities)
movements = np.zeros((n_cities, n_cities))

for i in range(n_cities):
    for j in range(n_cities):
        if i == j:
            movements[i, j] = 0
        else:
            Ci = cities[i]['population']
            Dj = cities[j]['population']
            Dij = euclidean_distance(cities[i]['location'], cities[j]['location'])
            movements[i, j] = gravity_model(Ci, Dj, Dij)

def animate(i):
    ax.clear()
    for idx, city in enumerate(cities):
        ax.scatter(*city['location'], s=city['population'] / 1000, label=city['name'], color=city['color'])
    for idx1 in range(n_cities):
        for idx2 in range(n_cities):
            if idx1 != idx2:
                coord1 = np.array(cities[idx1]['location'])
                coord2 = np.array(cities[idx2]['location'])
                n_dots = int(movements[idx1, idx2] / 1000000)  # Change this value to control the number of dots
                for dot in range(n_dots):
                    fraction = (i + dot * 100 / n_dots) % 100 / 100
                    dot_position = coord1 + np.subtract(coord2, coord1) * fraction
                    
                    # Add noise to the dot_position
                    noise = np.random.normal(scale=1, size=2)  # Change the scale value to control the noise magnitude
                    dot_position += noise
                    
                    # Set dot size to be proportional to the population movement
                    # dot_size = movements[idx1, idx2] / np.max(movements) * 500
                    
                    ax.scatter(*dot_position, s=10, alpha=0.5, color=cities[idx1]['color'])
    ax.set_title(f"Population Movement - Time Step: {i}")
    ax.set_xlim(-100, 1000)
    ax.set_ylim(-100, 1000)
    ax.legend()

fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, frames=100, interval=100, repeat=True)
plt.show()
