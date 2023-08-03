
import numpy as np
import pygame

cities = [
    {'name': 'City A', 'population': 1_000_000, 'location': (100, 100), 'color': (255, 0, 0)},
    {'name': 'City B', 'population': 2_000_000, 'location': (400, 400), 'color': (0, 0, 255)},
    {'name': 'City C', 'population': 1_500_000, 'location': (200, 600), 'color': (0, 255, 0)},
    {'name': 'City D', 'population': 500_000, 'location': (700, 100), 'color': (255, 165, 0)},
]

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def gravity_model(Ci, Dj, Dij, alpha=-2): # alpha is the gravity model exponent
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

# Initialize pygame and create a window
pygame.init()
WIDTH, HEIGHT = 800, 800  # dimensions of the window
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 10  # frames per second, higher values make the animation smoother

# Game loop
run = True
while run:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    # Clear the window
    win.fill((0, 0, 0))  # Fills the window with black color
    
    # Draw cities
    for city in cities:
        pygame.draw.circle(win, city['color'], city['location'], city['population']//100000) # Change the division value to control the city size

    # Draw movements
    for idx1 in range(n_cities):
        for idx2 in range(n_cities):
            if idx1 != idx2:
                coord1 = np.array(cities[idx1]['location'])
                coord2 = np.array(cities[idx2]['location'])
                n_dots = int(movements[idx1, idx2] / 1000000)  # Change this value to control the number of dots
                for dot in range(n_dots):
                    fraction = (pygame.time.get_ticks() + dot * 100 / n_dots) % 100 / 100
                    dot_position = coord1 + np.subtract(coord2, coord1) * fraction
                    
                    # Add noise to the dot_position
                    noise = np.random.normal(scale=3, size=2)  # Change the scale value to control the noise magnitude
                    dot_position += noise
                    
                    # Make sure dot_position is within the window
                    dot_position = np.clip(dot_position, 0, np.array([WIDTH-1, HEIGHT-1]))
                    
                    pygame.draw.circle(win, cities[idx1]['color'], dot_position.astype(int), 3) # Change the circle radius to control the dot size

    pygame.display.flip()  # Update the full display surface to the screen

pygame.quit()

# city size varies as the population moves in and out of the city
