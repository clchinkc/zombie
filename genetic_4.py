

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from genetic_4 import genetic_algorithm

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define fitness function for logistic regression problem
def fitness(chromosome):
    log_model = LogisticRegression()
    log_model.fit(X_train[:, chromosome], y_train)
    predictions = log_model.predict(X_test[:, chromosome])
    score = accuracy_score(y_test, predictions)
    return score

# Define parameters
pop_size = 100
chrom_len = X.shape[1]
n_generations = 100
mutation_rate = 0.1

if __name__ == "__main__":
    # Run genetic algorithm
    best_chromosome, best_fitness = genetic_algorithm(fitness, pop_size, chrom_len, n_generations, mutation_rate)
    print("Best chromosome:", '\n', '\n'.join(map(str, best_chromosome)))
    print("Best fitness:", '\n', best_fitness)

    # Train final model using the best chromosome in the last generation
    log_model = LogisticRegression()
    log_model.fit(X_train[:, best_chromosome[-1]], y_train)
    predictions = log_model.predict(X_test[:, best_chromosome[-1]])
    score = accuracy_score(y_test, predictions)

    # Print results
    print("Accuracy score using best chromosome:", score)
