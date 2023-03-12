# genetic algorithm search for discrete function optimization

import random


# Define the genetic algorithm function
def genetic_algorithm(fitness_func, initial_population, pop_size, elite_size, mutation_rate, generations):
    # Create the initial population
    population = initial_population

    # Loop through generations
    for i in range(generations):
        # Calculate the fitness of each individual in the population
        fitness_scores = [fitness_func(individual) for individual in population]

        # Select the top individuals (elite) to be passed on to the next generation
        elite_idx = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:elite_size]
        elite = [population[idx] for idx in elite_idx]

        # Create the new generation
        new_population = []

        # Fill the new population with elite individuals
        new_population.extend(elite)

        # Fill the rest of the new population through crossover
        num_crossover = pop_size - elite_size

        for j in range(num_crossover):
            # Select two parents from the population
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

            # Perform crossover to create a new child
            child = crossover(parent1, parent2)

            # Mutate the child
            mutate(child, mutation_rate)

            # Add the child to the new population
            new_population.append(child)

        # Set the population to be the new population
        population = new_population

        # Check if requirement is met
        best_fitness = max(fitness_scores)
        if best_fitness >= requirement:
            print(f"Requirement met after {i+1} generations.")
            return population

    # If the requirement is not met after the specified number of generations, return the final population
    print(f"Requirement not met after {generations} generations.")
    return population


# Define the crossover function (assuming binary representation)
def crossover(parent1, parent2):
    # Choose a random crossover point
    crossover_point = random.randint(1, len(parent1) - 1)

    # Perform crossover
    child = parent1[:crossover_point] + parent2[crossover_point:]

    return child


# Define the mutation function (assuming binary representation)
def mutate(child, mutation_rate):
    # Loop through each bit in the child
    for i in range(len(child)):
        # Check if this bit should be mutated
        if random.random() < mutation_rate:
            # Flip the bit
            child[i] = 1 - child[i]


# Example usage
def fitness_function(individual):
    # Define the fitness function (this is just a simple example)
    return sum(individual)

# Set the initial parameters
initial_population = [[random.randint(0, 1) for _ in range(10)] for _ in range(50)]
pop_size = 50
elite_size = 10
mutation_rate = 0.01
generations = 100
requirement = 10

# Run the genetic algorithm
new_population = genetic_algorithm(fitness_function, initial_population, pop_size, elite_size, mutation_rate, generations)
