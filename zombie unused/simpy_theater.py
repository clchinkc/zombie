import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import simpy

MAX_WAIT_TIME = 450  # Maximum wait time before a customer leaves
wait_times = []

class Theater(object):
    def __init__(self, env, num_cashiers, num_servers, num_ushers, peak_hours, special_event):
        self.env = env
        self.cashier = simpy.Resource(env, num_cashiers)
        self.server = simpy.Resource(env, num_servers)
        self.usher = simpy.Resource(env, num_ushers)
        self.peak_hours = peak_hours
        self.special_event = special_event
        self.cashier_availability = 1.0  # 100% availability initially
        self.dissatisfied_customers_count = 0

    def purchase_ticket(self, moviegoer):
        yield self.env.timeout(random.randint(1, 3) if not self.peak_hours else random.randint(2, 5))
        print(f"Moviegoer {moviegoer} purchased a ticket.")

    def check_ticket(self, moviegoer):
        yield self.env.timeout(3 / 60)
        print(f"Moviegoer {moviegoer} had their ticket checked.")

    def sell_food(self, moviegoer):
        yield self.env.timeout(random.randint(1, 5) if not self.special_event else random.randint(3, 7))
        print(f"Moviegoer {moviegoer} bought some food.")

    def cashier_available(self):
        return random.random() < self.cashier_availability

def go_to_movies(env, moviegoer, theater):
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        if not theater.cashier_available():
            print(f"Moviegoer {moviegoer} left due to no available cashier.")
            theater.dissatisfied_customers_count += 1
            return

        yield env.process(theater.purchase_ticket(moviegoer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

    if random.random() < 0.5:
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(moviegoer))

    wait_duration = env.now - arrival_time
    if wait_duration >= MAX_WAIT_TIME:
        print(f"Moviegoer {moviegoer} left due to long wait time.")
        theater.dissatisfied_customers_count += 1
    else:
        wait_times.append(wait_duration)

def run_theater(env, theater):
    moviegoer = 0
    while True:
        yield env.timeout(np.random.poisson(1) if not theater.special_event else np.random.poisson(3))
        moviegoer += 1
        env.process(go_to_movies(env, moviegoer, theater))

def adjust_cashier_availability(env, theater, peak_hours):
    while True:
        current_hour = env.now // 60
        theater.cashier_availability = 0.5 if current_hour in peak_hours else 1.0
        yield env.timeout(60)  # Check every hour

def calculate_wait_time(wait_times):
    average_wait = statistics.mean(wait_times)
    std_dev = statistics.stdev(wait_times)
    return average_wait, std_dev

def plot_wait_times(wait_times):
    plt.hist(wait_times, bins=10, edgecolor='black')
    plt.xlabel('Wait Time (Seconds)')
    plt.ylabel('Number of Customers')
    plt.title('Distribution of Customer Wait Times')
    plt.show()

def main():
    random.seed(42)

    num_cashiers, num_servers, num_ushers = 1, 1, 1
    peak_hours = False
    special_event = False

    env = simpy.Environment()
    theater = Theater(env, num_cashiers, num_servers, num_ushers, peak_hours, special_event)

    env.process(run_theater(env, theater))
    env.process(adjust_cashier_availability(env, theater, [18, 19, 20]))  # Example peak hours
    env.run(until=1000)

    total_moviegoers = len(wait_times) + theater.dissatisfied_customers_count
    print(f"Total moviegoers processed: {total_moviegoers}")
    print(f"Moviegoers who waited less than {MAX_WAIT_TIME} seconds: {len(wait_times)}")
    print(f"Dissatisfied Customers: {theater.dissatisfied_customers_count}")

    if wait_times:
        average_wait, std_dev = calculate_wait_time(wait_times)
        print(f"Average wait time: {average_wait:.2f} seconds")
        print(f"Standard Deviation of wait times: {std_dev:.2f} seconds")
    else:
        print("No data on wait times to calculate average and standard deviation.")

    plot_wait_times(wait_times)

if __name__ == '__main__':
    main()

"""
What's Left to Be Updated
Detailed Implementation of Visualization: The example provided is basic. You can create more sophisticated visualizations based on your specific needs, like time-series plots, bar graphs for resource utilization, etc.

In-Depth Scenario Analysis: Implement and run various scenarios to analyze different aspects of the theater operation, like peak vs. off-peak hours, special events, and varying staff levels.

Further Refinement of Customer Behaviors and Satisfaction Metrics: You can add more complex customer behavior models and refine satisfaction metrics to be more representative of real-world scenarios.

Advanced Statistical Analysis: If needed, you can incorporate more advanced statistical analysis to better understand the simulation results.

User Interaction for Scenario Selection: If desired, you can add an interface for users to select different scenarios and parameters without modifying the code.
"""