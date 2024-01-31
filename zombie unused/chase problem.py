
import numpy as np
import pygame
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tf_agents.agents import CategoricalDqnAgent, PPOKLPenaltyAgent, ReinforceAgent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import (
    actor_distribution_network,
    categorical_q_network,
    q_network,
    value_network,
)
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.train.utils import strategy_utils, train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class Runner:
    def __init__(self, position, grid_size):
        self.position = position
        self._grid_size = grid_size

    def move(self, direction, is_valid_move):
        new_position = self.position.copy()
        if direction == 0:   # up
            new_position[0] -= 1
        elif direction == 1: # right
            new_position[1] += 1
        elif direction == 2: # down
            new_position[0] += 1
        elif direction == 3: # left
            new_position[1] -= 1
        if is_valid_move(new_position) == True:
            self.position = new_position

class Chaser:
    def __init__(self, position):
        self.position = position

    def move_towards_runner(self, runner_position, is_valid_move):
        move_direction = np.argmax(np.abs(runner_position - self.position))
        direction_sign = np.sign(runner_position[move_direction] - self.position[move_direction])
        new_position = self.position.copy()
        new_position[move_direction] += direction_sign
        if is_valid_move(new_position) == True:
            self.position = new_position

class Obstacle:
    def __init__(self, position):
        self.position = position

class ChaseEnvironment(py_environment.PyEnvironment):
    def __init__(self, grid_size, num_obstacles):
        self._grid_size = grid_size
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(grid_size, grid_size, 3), dtype=np.float32, minimum=0., maximum=1., name='observation')
        self._runner = Runner(self._init_position(), grid_size)
        self._chaser = Chaser(self._init_position())
        self._obstacles = [Obstacle(self._init_position()) for _ in range(num_obstacles)]
        self._state = self._create_state()

        # Define the action distribution spec
        self._action_distribution_spec = tfp.distributions.Categorical(logits=tf.zeros((self._action_spec.maximum + 1,)))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _init_position(self):
        return np.random.randint(0, self._grid_size, size=(2,))
    
    def _reset_position(self):
        position = self._init_position()
        while not self.is_valid_position(position):
            position = self._init_position()
        return position
    
    def is_valid_position(self, position):
        is_inside_grid = 0 <= position[0] < self._grid_size and 0 <= position[1] < self._grid_size
        not_occupied =  not any(np.array_equal(position, obs.position) for obs in self._obstacles)
                        # not np.array_equal(position, self._runner.position) and \
                        # not np.array_equal(position, self._chaser.position)

        return is_inside_grid and not_occupied

    def _create_state(self):
        state = np.zeros((self._grid_size, self._grid_size, 3), dtype=np.float32)
        state[tuple(self._runner.position), 0] = 1.
        state[tuple(self._chaser.position), 1] = 1.
        for obs in self._obstacles:
            state[tuple(obs.position), 2] = 1.
        return state

    def _is_done(self):
        return np.array_equal(self._runner.position, self._chaser.position)

    def _reward(self):
        return np.linalg.norm(self._runner.position - self._chaser.position)
        # return -np.linalg.norm(self._runner_position - self._chaser_position) for chaser
        # return 1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for runner for smoothly approach
        # return -1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for chaser for smoothly avoid

    def _reset(self):
        self._runner.position = self._reset_position()
        self._chaser.position = self._reset_position()
        self._state = self._create_state()
        return ts.restart(self._state)

    def _step(self, action):
        if self._is_done():
            return self._reset()

        self._runner.move(action, self.is_valid_position)
        self._chaser.move_towards_runner(self._runner.position, self.is_valid_position)
        self._state = self._create_state()

        if self._is_done():
            return ts.termination(self._state, self._reward())
        else:
            return ts.transition(self._state, reward=self._reward(), discount=0.999)

    def _render(self):
        # if there is no screen, create one
        if not hasattr(self, '_screen'):
            # Create the display surface
            cell_size = 50
            width, height = self._grid_size * cell_size, self._grid_size * cell_size
            self.screen = pygame.display.set_mode((width, height))
            self.cell_size = cell_size

            # Create the grid rects
            self.grid_rects = [pygame.Rect(i, j, cell_size, cell_size) for j in range(0, height, cell_size) for i in range(0, width, cell_size)]

            # Create runner, chaser, and obstacle rects
            self.runner_rect = pygame.Rect(0, 0, cell_size, cell_size)
            self.chaser_rect = pygame.Rect(0, 0, cell_size, cell_size)
            self.obstacle_rects = [pygame.Rect(0, 0, cell_size, cell_size) for _ in range(num_obstacles)]
        
        self.screen.fill((0, 0, 0))

        # Draw the grid
        for rect in self.grid_rects:
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Update and draw the runner (blue) and the chaser (red)
        self.runner_rect.topleft = (self._runner.position[1] * self.cell_size, self._runner.position[0] * self.cell_size)
        self.chaser_rect.topleft = (self._chaser.position[1] * self.cell_size, self._chaser.position[0] * self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), self.runner_rect)
        pygame.draw.rect(self.screen, (255, 0, 0), self.chaser_rect)

        # Update and draw the obstacles (gray)
        for i, obstacle in enumerate(self._obstacles):
            self.obstacle_rects[i].topleft = (obstacle.position[1] * self.cell_size, obstacle.position[0] * self.cell_size)
            pygame.draw.rect(self.screen, (128, 128, 128), self.obstacle_rects[i])

        pygame.display.flip()

num_iterations = 2000
initial_collect_steps = 1000
collect_steps_per_iteration = 2
replay_buffer_max_length = 10000
batch_size = 64
log_interval = 100
num_eval_episodes = 10
eval_interval = 10000

grid_size = 10
num_obstacles = 3
env = ChaseEnvironment(grid_size=grid_size, num_obstacles=num_obstacles)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

initial_learning_rate = 0.001
alpha = 0.5
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=num_iterations,
    alpha=alpha
)

decay_epsilon_greedy = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.9,
    decay_steps=num_iterations,
    end_learning_rate=0.1
)

num_steps_update = 5

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

def create_dqn_agent(lr_schedule, train_env, decay_epsilon_greedy=decay_epsilon_greedy, num_steps_update=num_steps_update):

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=((16, 3, 1), (32, 3, 1)),
        fc_layer_params=(10,),
        dropout_layer_params=(0.1,),
        activation_fn=tf.keras.activations.elu,
        )

    train_step_counter = train_utils.create_train_step()

    tf_agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule(train_step_counter)),
        epsilon_greedy=lambda: decay_epsilon_greedy(train_step_counter),
        n_step_update=num_steps_update,
        td_errors_loss_fn=common.element_wise_squared_loss, # tf.losses.Huber(reduction="none")
        train_step_counter=train_step_counter)

    tf_agent.initialize()
    return train_step_counter, tf_agent

def create_ddpn_agent(learning_rate, train_env):
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=((16, 3, 1), (32, 3, 1)),
        fc_layer_params=(10,),
        dropout_layer_params=(0.1,),
        activation_fn=tf.keras.activations.elu,
        )
    
    train_step_counter = train_utils.create_train_step()
    
    tf_agent = DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule(train_step_counter)),
        epsilon_greedy=lambda: decay_epsilon_greedy(train_step_counter),
        n_step_update=num_steps_update,
        td_errors_loss_fn=common.element_wise_squared_loss, # tf.losses.Huber(reduction="none")
        train_step_counter=train_step_counter)
    
    tf_agent.initialize()
    
    return train_step_counter, tf_agent

def create_categorical_dqn_agent(learning_rate, train_env, decay_epsilon_greedy=decay_epsilon_greedy, num_steps_update=num_steps_update):

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=51,
        conv_layer_params=((16, 3, 1), (32, 3, 1)),
        fc_layer_params=(10,),
        activation_fn=tf.keras.activations.elu,
        )

    train_step_counter = train_utils.create_train_step()
    
    def categorical_huber_loss(target, pred):
        # Use the huber loss to compute the element-wise loss for the target and pred tensors.
        elementwise_loss = tf.losses.Huber(target, pred, reduction=tf.losses.Reduction.NONE)

        # Compute the categorical loss by summing the element-wise loss over the atoms dimension
        # and multiplying by the atom delta.
        return tf.reduce_sum(elementwise_loss, axis=-1) * categorical_q_net.atom_delta

    tf_agent = CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule(train_step_counter)),
        epsilon_greedy=lambda: decay_epsilon_greedy(train_step_counter),
        n_step_update=num_steps_update,
        target_update_tau=0.05,
        target_update_period=1,
        td_errors_loss_fn=categorical_huber_loss,
        train_step_counter=train_step_counter)

    tf_agent.initialize()
    return train_step_counter, tf_agent

def create_reinforce_agent(learning_rate, train_env):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(100,),
    )
    
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(100,),
    )

    train_step_counter = train_utils.create_train_step()

    tf_agent = ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        value_network=value_net,
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule(train_step_counter)),
        use_advantage_loss=True,
        value_estimation_loss_coef=0.5,
        gamma=0.99,
        normalize_returns=True,
        gradient_clipping=0.5,
        entropy_regularization=0.2,
        train_step_counter=train_step_counter,
    )

    tf_agent.initialize()

    return train_step_counter, tf_agent

def create_ppo_agent(learning_rate, train_env):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(100,),
    )

    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(100,),
    )

    train_step_counter = train_utils.create_train_step()

    tf_agent = PPOKLPenaltyAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule(train_step_counter)),
        num_epochs=1,
        initial_adaptive_kl_beta=1.0,
        adaptive_kl_target=0.01,
        adaptive_kl_tolerance=0.5,
        kl_cutoff_coef=0.0,
        kl_cutoff_factor=0.0,
        use_gae=True,
        normalize_rewards=True,
        entropy_regularization=0.1,
        train_step_counter=train_step_counter,
    )

    tf_agent.initialize()

    return train_step_counter, tf_agent


with strategy.scope():
    # train_step_counter, tf_agent = create_dqn_agent(lr_schedule, train_env)
    train_step_counter, tf_agent = create_ddpn_agent(lr_schedule, train_env)
    # train_step_counter, tf_agent = create_categorical_dqn_agent(lr_schedule, train_env)
    # train_step_counter, tf_agent = create_reinforce_agent(lr_schedule, train_env)
    # train_step_counter, tf_agent = create_ppo_agent(lr_schedule, train_env)

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

# collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)
collect_data(eval_env, tf_agent.collect_policy, replay_buffer, steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=4,
    sample_batch_size=batch_size,
    num_steps=num_steps_update + 1
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

train_checkpointer.initialize_or_restore()

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

# def eval_agent(environment, policy, num_episodes=num_eval_episodes):
#     metrics = [
#         tf_metrics.AverageReturnMetric(buffer_size=num_episodes),
#         tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes)
#     ]
#     results = metric_utils.eager_compute(
#         metrics=metrics,
#         environment=environment,
#         policy=policy,
#         num_episodes=num_episodes
#     )
#     return results['AverageReturn'], results['AverageEpisodeLength']

def eval_agent(environment, policy, num_episodes=num_eval_episodes):
    average_return_metric = tf_metrics.AverageReturnMetric(buffer_size=num_episodes)
    average_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes)
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        environment,
        policy,
        observers=[average_return_metric, average_episode_length_metric],
        num_episodes=num_episodes
    )
    driver.run(maximum_iterations=1000)
    avg_return = average_return_metric.result().numpy()
    avg_length = average_episode_length_metric.result().numpy()
    return avg_return, avg_length

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
            pygame.time.wait(1000)
            
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            
            # Render the environment after taking a step
            environment.pyenv.envs[0]._render()
            
    # Quit Pygame after the evaluation is done
    pygame.time.wait(1000)
    pygame.quit()

if __name__ == '__main__':
    train_agent(num_iterations, collect_steps_per_iteration, log_interval, num_eval_episodes, eval_interval, train_env, eval_env, train_step_counter, tf_agent, replay_buffer, iterator, train_checkpointer, collect_step, eval_agent)
    num_test_episodes = 10
    avg_return, avg_length = eval_agent(eval_env, tf_agent.policy, num_test_episodes)
    print('Average Return = {0}, Average Episode Length = {1}'.format(avg_return, avg_length))
    # sim_agent(eval_env, tf_agent.policy, num_test_episodes)
    # print('Simulation done!')


# https://github.com/tensorflow/agents/blob/master/tf_agents/agents/dqn/examples/v2/train_eval.py
# Reward = overall enemy casualties
# https://github.com/christianhidber/easyagents api across libraries
# each object has a draw method, call all draw method in the environment
# update update part of the environment and flip update the whole screen
# Dueling dqn (there is an unmerged pull request in the tf_agents repo)
# Prioritized experience replay (https://kenneth-schroeder-dev.medium.com/prioritized-experience-replay-with-tf-agents-3fa2498f411a, https://gist.github.com/Kenneth-Schroeder/6dbdd4e165331164e0d9dcc2355698e2)
# Soft actor critic in tf_agents
# TD3 in tf_agents
# Rainbow DQN
# Independent Q-Learning
# Multi-Agent Deep Q-Network
# Q-learning with social learning
# n-step value estimates may use rnns.
# AlphaGo Zero: adding a policy head rather than having policy and value in separate nets led to a huge gain, combining l2 and cross entropy loss
# https://blog.csdn.net/wxc971231/article/details/127567643

# https://github.com/Lostefra/ReinforcementLearningToy/tree/main format
# https://github.com/telkhir/Deep-RL format
# https://github.com/priontu/Atari_DemonAttack_Gameplay_with_Reinforcement_Learning_using_TF_Agents format
# https://github.com/marvinschmitt/DeepLearning-Bomberman MCTS + DQN
# https://github.com/marload/DeepRL-TensorFlow2 tensorflow from scratch
# https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a tensorflow from scratch
# https://github.com/nslyubaykin/relax pytorch
# https://github.com/nslyubaykin/rainbow_for_2048 pytorch
# https://github.com/nslyubaykin/relax_rainbow_dqn_example pytorch
# https://github.com/cyoon1729/RLcycle pytorch
# https://github.com/davide97l/Rainbow pytorch rainbow
# https://github.com/Curt-Park/rainbow-is-all-you-need pytorch
# https://github.com/deepmind/dqn_zoo jax
# https://zhuanlan.zhihu.com/p/220510418
# https://github.com/willi-menapace/atari_reinforcement_learning pytorch rainbow
# https://github.com/chucnorrisful/dqn keras-rl
# https://towardsdatascience.com/rainbow-dqn-the-best-reinforcement-learning-has-to-offer-166cb8ed2f86 rainbow
# https://zhuanlan.zhihu.com/p/261322143 rainbow dqn
# 以下是每個擴展方法如何提高數據效率和最終性能的詳細解釋：
# 1. Double Q-learning：傳統的Q-learning算法容易高估某些動作的值，進而導致學習到低質量的策略。Double Q-learning通過使用兩個獨立的Q-networks來解決這個問題，從而減少了高估值的影響。這種方法提高了算法學習到高質量策略的速度和效率。
# 2. Prioritized Experience Replay：傳統的經驗回放方法是按照時間順序隨機選取存儲在緩存中的經驗來進行訓練。但是，某些重要的經驗可能對算法性能有更大的貢獻。Prioritized Experience Replay通過根據其對算法性能貢獻大小來選擇重要性較高的經驗，從而提高了數據效率和最終性能。
# 3. Dueling Network Architectures：Dueling Network Architectures將Q-networks分成兩部分：一部分用於估計動作價值，另一部分用於估計基本價值。這種方法使算法能夠更好地學習到不同動作之間的差異，從而提高了最終性能。
# 4. Multi-step Learning：傳統的Q-learning算法僅考慮當前狀態和下一個狀態之間的轉移。Multi-step Learning通過考慮多個連續狀態之間的轉移，從而使算法能夠更好地利用長期的時間關聯性，從而提高了數據效率和最終性能。
# 5. Distributional RL：傳統的Q-learning算法僅考慮每個動作的期望回報值。Distributional RL通過估計每個動作的回報分佈，從而提高了算法對不同回報值之間差異的感知能力。這種方法使算法能夠更好地學習到不同動作之間的差異，從而提高了最終性能。
# 6. Noisy Nets：Noisy Nets通過向神經網絡中添加隨機噪聲，從而使得神經網絡更容易探索新的策略。這種方法提高了算法學習到高質量策略的速度和效率。
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/bandits
# https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab
# https://www.tensorflow.org/decision_forests/tutorials/model_composition_colab
# https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression
# https://worldmodels.github.io/
# https://github.com/ctallec/world-models
# https://blog.otoro.net/2018/06/09/world-models-experiments/
# https://github.com/hardmaru/WorldModelsExperiments
# https://github.com/topics/world-models
# https://telefonicatech.com/en/blog/realistic-worlds-procedural-generation-artificial-intelligence-video-games
# https://arxiv.org/abs/2209.00588
# https://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf
# https://arxiv.org/pdf/1511.09249.pdf
# https://github.com/zhongwen/predictron

# Universal Value Function Approximators (UVFAs)
# https://proceedings.mlr.press/v37/schaul15.pdf
# https://github.com/rllabmcgill/rlcourse-march-17-hugobb

# intrinsic motivation, curiosity-driven exploration, count-based exploration

# VIME (Houthooft et al., 2016)
# count-based exploration (Ostrovski et al., 2017)
# bootstrapped DQN (Osband et al., 2016)

# https://en.m.wikipedia.org/wiki/Multi-armed_bandit

# 添加更多學習信號
# Hindsight Experience Replay
# https://arxiv.org/abs/1707.01495
# Auxiliary Tasks
# https://arxiv.org/abs/1611.05397

# 基於模型的學習
# https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=711FEF6BA26BBF98C28BC111B26F8761?doi=10.1.1.48.6005&rep=rep1&type=pdf
# https://zhuanlan.zhihu.com/p/524200581
# https://zhuanlan.zhihu.com/p/524200581

# Difficulties in Deep Reinforcement Learning
# https://arxiv.org/abs/1709.06560
# https://openai.com/research/faulty-reward-functions

# Monte Carlo Tree Search for Atari
# https://papers.nips.cc/paper_files/paper/2014/hash/8bb88f80d334b1869781beb89f7b73be-Abstract.html

# 逆向強化學習和模仿學習
# https://ai.stanford.edu/~ang/papers/icml00-irl.pdf
# http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
# https://arxiv.org/pdf/1603.00448.pdf
# https://papers.nips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html
# https://arxiv.org/abs/1711.02827
# http://proceedings.mlr.press/v78/bajcsy17a/bajcsy17a.pdf

# https://towardsdatascience.com/hyperbolic-deep-reinforcement-learning-b2de787cf2f7

# Multi-Agent PPO
# https://github.com/marlbenchmark/on-policy