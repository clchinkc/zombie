import math
import random

import pygame


class City:
    def __init__(self, name, population, location, color):
        self.name = name
        self.population = population
        self.location = location
        self.color = color

    def draw(self, screen):
        city_size = int(math.sqrt(self.population) / 100) if self.population > 0 else 0
        pygame.draw.circle(screen, self.color, self.location, city_size)


# Constants
WIDTH, HEIGHT = 800, 600
FPS = 10
DOT_SPEED = 1.0 / 20
MOVEMENT_SCALE_FACTOR = 10000
RANDOM_RANGE = 10
ALPHA = 3


def initialize_cities():
    return [
        City("City A", 1000000., (100, 150), (255, 0, 0)),
        City("City B", 1500000., (500, 150), (0, 255, 0)),
        City("City C", 500000., (300, 400), (0, 0, 255)),
        City("City D", 800000., (700, 300), (255, 255, 0))
    ]


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def gravity_model(pop1, pop2, distance, alpha=ALPHA):
    try:
        return (pop1 * pop2) / (distance ** alpha)
    except ZeroDivisionError:
        return 0


def interpolate(start, end, factor):
    return start[0] + factor * (end[0] - start[0]), start[1] + factor * (end[1] - start[1])


def randomize_position(position, range=RANDOM_RANGE):
    return position[0] + random.randint(-range, range), position[1] + random.randint(-range, range)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('City Movement Simulation')
    clock = pygame.time.Clock()

    cities = initialize_cities()
    dots = []

    running = True
    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i, city_from in enumerate(cities):
            for j, city_to in enumerate(cities):
                if i != j:
                    dist = euclidean_distance(city_from.location, city_to.location)
                    move = gravity_model(city_from.population, city_to.population, dist)
                    city_from.population = max(city_from.population - move, 0)
                    city_to.population += move

                    scaled_movement = int(move / MOVEMENT_SCALE_FACTOR)
                    for _ in range(scaled_movement):
                        random_start = randomize_position(city_from.location)
                        random_end = randomize_position(city_to.location)
                        dots.append({'from': random_start, 'to': random_end, 'progress': 0, 'color': city_from.color})

        for city in cities:
            city.draw(screen)

        new_dots = []
        for dot in dots:
            dot['progress'] += DOT_SPEED
            if dot['progress'] < 1.0:
                new_dots.append(dot)
                dot_position = interpolate(dot['from'], dot['to'], dot['progress'])
                pygame.draw.circle(screen, dot['color'], (int(dot_position[0]), int(dot_position[1])), 2)

        dots = new_dots
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
