

import numpy as np
import pygame
import tensorflow as tf
from matplotlib import pyplot as plt
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import (
    actor_distribution_network,
    normal_projection_network,
    q_network,
    value_network,
)
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class ChaseEnvironment(py_environment.PyEnvironment):

    def __init__(self, grid_size):
        self._grid_size = grid_size
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(grid_size, grid_size, 2), dtype=np.float32, minimum=0., maximum=1., name='observation')
        self._runner_position = self._reset_position()
        self._chaser_position = self._reset_position()
        self._state = self._create_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset_position(self):
        return np.random.randint(0, self._grid_size, size=(2,))

    def _create_state(self):
        state = np.zeros((self._grid_size, self._grid_size, 2), dtype=np.float32)
        state[tuple(self._runner_position), 0] = 1.
        state[tuple(self._chaser_position), 1] = 1.
        return state

    def _take_action(self, action):
        new_position = self._runner_position.copy()
        if action == 0:   # up
            new_position[0] = np.clip(new_position[0] - 1, 0, self._grid_size - 1)
        elif action == 1: # right
            new_position[1] = np.clip(new_position[1] + 1, 0, self._grid_size - 1)
        elif action == 2: # down
            new_position[0] = np.clip(new_position[0] + 1, 0, self._grid_size - 1)
        elif action == 3: # left
            new_position[1] = np.clip(new_position[1] - 1, 0, self._grid_size - 1)
        return new_position

    def _move_chaser(self):
        move_direction = np.argmax(np.abs(self._runner_position - self._chaser_position))
        direction_sign = np.sign(self._runner_position[move_direction] - self._chaser_position[move_direction])
        self._chaser_position[move_direction] += direction_sign

    def _is_done(self):
        return np.array_equal(self._runner_position, self._chaser_position)

    def _reward(self):
        return np.linalg.norm(self._runner_position - self._chaser_position)
        # return -np.linalg.norm(self._runner_position - self._chaser_position) for chaser
        # return 1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for runner for smoothly approach
        # return -1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for chaser for smoothly avoid

    def _reset(self):
        self._runner_position = self._reset_position()
        self._chaser_position = self._reset_position()
        self._state = self._create_state()
        return ts.restart(self._state)

    def _step(self, action):
        if self._is_done():
            return self.reset()

        self._runner_position = self._take_action(action)
        self._move_chaser()
        self._state = self._create_state()

        if self._is_done():
            return ts.termination(self._state, self._reward())
        else:
            return ts.transition(self._state, reward=self._reward(), discount=0.999)

    def _render(self):
        cell_size = 50
        width, height = self._grid_size * cell_size, self._grid_size * cell_size
        screen = pygame.display.set_mode((width, height))
        screen.fill((0, 0, 0))

        # Draw the grid
        for i in range(0, width, cell_size):
            for j in range(0, height, cell_size):
                rect = pygame.Rect(i, j, cell_size, cell_size)
                pygame.draw.rect(screen, (255, 255, 255), rect, 1)

        # Draw the runner (blue) and the chaser (red)
        runner_rect = pygame.Rect(self._runner_position[1] * cell_size, self._runner_position[0] * cell_size, cell_size, cell_size)
        chaser_rect = pygame.Rect(self._chaser_position[1] * cell_size, self._chaser_position[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 0, 255), runner_rect)
        pygame.draw.rect(screen, (255, 0, 0), chaser_rect)

        pygame.display.flip()

num_iterations = 20
initial_collect_steps = 10
collect_steps_per_iteration = 1
replay_buffer_max_length = 10000
batch_size = 64
learning_rate = 1e-3
log_interval = 5
num_eval_episodes = 1
eval_interval = 1000

grid_size = 10
env = ChaseEnvironment(grid_size)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

def create_dqn_agent(learning_rate, train_env):
    conv_layer_params = ((16, 3, 1), (32, 3, 1))
    fc_layer_params = (10,)
    q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
    scale=2.0, mode='fan_in', distribution='truncated_normal')
    )

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    train_step_counter = tf.Variable(0)

    tf_agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss, # tf.losses.Huber(reduction="none")
    train_step_counter=train_step_counter)

    tf_agent.initialize()
    return train_step_counter, tf_agent

def create_ppo_agent(learning_rate, train_env):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(256,),
        #continuous_projection_net=(normal_projection_network.NormalProjectionNetwork,)
    )

    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(256,)
    )

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    tf_agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.001,
        #importance_ratio_clipping=0.2,
        #lambda_value=0.95,
        #discount_factor=0.99,
        num_epochs=5,
        debug_summaries=False,
        #summarize_grads_and_vars=False,
        train_step_counter=train_step_counter,
    )

    tf_agent.initialize()
    return train_step_counter, tf_agent


train_step_counter, tf_agent = create_dqn_agent(learning_rate, train_env)
# train_step_counter, tf_agent = create_ppo_agent(learning_rate, train_env)

def create_replay_buffer(replay_buffer_max_length, train_env, tf_agent):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length,
    device='gpu:0'
    )

    return replay_buffer

replay_buffer = create_replay_buffer(replay_buffer_max_length, train_env, tf_agent)

def collect_data(env, policy, buffer, steps):
    driver = dynamic_step_driver.DynamicStepDriver(
        env,
        policy,
        observers=[buffer.add_batch],
        num_steps=steps)
    driver.run()

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2,
    ).prefetch(tf.data.experimental.AUTOTUNE)

iterator = iter(dataset)

tf_agent.train = common.function(tf_agent.train)

train_checkpointer = common.Checkpointer(
    ckpt_dir='checkpoint',
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

def eval_agent(environment, policy, num_episodes=num_eval_episodes):
    metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes)
    ]
    results = metric_utils.compute(
        metrics=metrics,
        environment=environment,
        policy=policy,
        num_episodes=num_episodes
    )
    return results['AverageReturn'], results['AverageEpisodeLength']

# def eval_agent(environment, policy, num_episodes=num_eval_episodes):
#     average_return_metric = tf_metrics.AverageReturnMetric(buffer_size=num_episodes)
#     average_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes)
#     driver = dynamic_episode_driver.DynamicEpisodeDriver(
#         environment,
#         policy,
#         observers=[average_return_metric, average_episode_length_metric],
#         num_episodes=num_episodes
#     )
#     driver.run(maximum_iterations=1000)
#     environment.render()
#     avg_return = average_return_metric.result().numpy()
#     avg_length = average_episode_length_metric.result().numpy()
#     return avg_return, avg_length

def train_agent(num_iterations, collect_steps_per_iteration, log_interval, num_eval_episodes, eval_interval, train_env, eval_env, train_step_counter, tf_agent, replay_buffer, iterator, train_checkpointer, collect_step, compute_avg_return):
    
    returns = []
    
    for _ in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, tf_agent.collect_policy, replay_buffer)

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return, avg_length = eval_agent(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Episode Length = {2}'.format(step, avg_return, avg_length))
            train_checkpointer.save(train_step_counter.numpy())
            print('Saved checkpoint for step {0}'.format(step))
            returns.append(avg_return)

    if returns:
        # Plot the results
        plt.plot(returns)
        plt.xlabel('Step')
        plt.ylabel('Average Return')
        plt.title('Training Progress')
        plt.show()

def sim_agent(environment, policy, num_episodes=num_eval_episodes):
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Chase Environment")
    
    for _ in range(num_episodes):
        time_step = environment.reset()
        environment.pyenv.envs[0]._render()  # Render the initial state

        while not time_step.is_last():
            pygame.time.wait(1000)  # Adjust the delay to control the speed of the visualization
            
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            
            # Render the environment after taking a step
            environment.pyenv.envs[0]._render()
            
    # Quit Pygame after the evaluation is done
    pygame.time.wait(1000)  # Adjust the delay to control the speed of the visualization
    pygame.quit()

if __name__ == '__main__':
    train_agent(num_iterations, collect_steps_per_iteration, log_interval, num_eval_episodes, eval_interval, train_env, eval_env, train_step_counter, tf_agent, replay_buffer, iterator, train_checkpointer, collect_step, eval_agent)
    num_test_episodes = 2
    avg_return, avg_length = eval_agent(eval_env, tf_agent.policy, num_test_episodes)
    print('Average Return = {0}, Average Episode Length = {1}'.format(avg_return, avg_length))
    sim_agent(eval_env, tf_agent.policy, num_test_episodes)
    print('Simulation done!')

# https://www.tensorflow.org/agents/tutorials/6_reinforce_tutorial
# each object has a draw method, call all draw method in the environment
# update update part of the environment and flip update the whole screen
# dqn agent change to reinforce agent and ppo agent
