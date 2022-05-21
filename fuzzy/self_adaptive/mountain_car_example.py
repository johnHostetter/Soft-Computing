import os
import gym
import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy.frl.fql.fql import FQLModel
from fuzzy.frl.cfql.cfql import CFQLModel
from fuzzy.frl.fql.fis import InputStateVariable, Trapeziums, Build

from fuzzy.self_adaptive.make_rules_and_terms import unsupervised

from d3rlpy.algos import DiscreteCQL
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split

GLOBAL_SEED = 1
LOCAL_SEED = 42
np.random.seed(GLOBAL_SEED)

def evaluate_on_environment(env, n_trials=100, epsilon=0.0, render=False):
    """ Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env (gym.Env): gym-styled environment.
        n_trials (int): the number of trials.
        epsilon (float): noise factor for epsilon-greedy policy.
        render (bool): flag to render environment.

    Returns:
        callable: scoerer function.


    """

    def scorer(algo, *args):
        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.std(episode_rewards)

    return scorer


class Agent:
    def __init__(self, model):
        self.model = model

    def get_initial_action(self, state):
        try:
            return self.model.get_initial_action(state)
        except AttributeError:
            return self.model.get_action(state)

    def get_action(self, state):
        return self.model.get_action(state)

    def learn(self, state, reward):
        return self.model.run(state, reward)


# Define membership functions for MountainCar problems
def get_fis_env():
    p = InputStateVariable(Trapeziums(-1.2, -1.2, -1.2, -0.775),
                           Trapeziums(-1.2, -0.775, -0.775, -0.35),
                           Trapeziums(-0.775, -0.35, -0.35, 0.075),
                           Trapeziums(-0.35, 0.075, 0.075, 0.5),
                           Trapeziums(0.075, 0.5, 0.5, 0.5))
    v = InputStateVariable(Trapeziums(-0.07, -0.07, -0.07, -0.035),
                           Trapeziums(-0.07, -0.035, -0.035, 0.),
                           Trapeziums(-0.035, 0., 0., 0.035),
                           Trapeziums(0., 0.035, 0.035, 0.07),
                           Trapeziums(0.035, 0.035, 0.035, 0.07))
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    fis = Build(p, v)
    env.seed(LOCAL_SEED)
    return env, fis


def play_mountain_car(model, max_eps=100, render=False):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)

    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    visited_states = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            visited_states.append(state_value)
            action = model.get_action(state_value)
            rewards.append(r)
            mean_reward = np.mean(rewards[-50:])
            best_mean_rewards.append(mean_reward)
            if len(best_mean_rewards) > 2:
                epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
            else:
                epsilon = 0
            try:
                print('EPS=', iteration, ' reward=', r,
                      ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            except AttributeError:
                print('EPS=', iteration, ' reward=', r,
                      ' epsilon=', 0.0, ' best mean eps=', epsilon)
            iteration += 1
            r = 0

        # render the environment for the last couple episodes
        if render and iteration + 1 > (max_eps - 6):
            env.render()

        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        visited_states.append(state_value)
        trajectories.append((prev_state, action, reward, state_value, done))
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = model.get_action(state_value)

        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print(model.q_table)
    try:
        print('Epsilon=', model.ee_rate)
    except AttributeError:
        print('Epsilon=', 0.0)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()

    env.close()

    return model, np.array(visited_states), trajectories, rewards


def random_play_mountain_car(model=None, max_eps=500):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)

    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            action = env.action_space.sample()
            rewards.append(r)
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', 1.0, ' best mean eps=', 1.0)
            iteration += 1
            r = 0

        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        # trajectories.append((prev_state, action, reward, state_value, done))
        trajectories.append((prev_state, action, state_value, reward, done))  # this is how my algorithm impl. expects data
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = env.action_space.sample()
        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print('Epsilon=', 1.0)
    return model, trajectories, rewards


def train_env(model=None, max_eps=500):
    env, fis = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

    # Create Model
    if model is None:
        model = FQLModel(gamma=0.99,
                         alpha=0.1,
                         ee_rate=1.,
                         action_set_length=action_set_length,
                         fis=fis)
    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            action = model.get_initial_action(state_value)
            rewards.append(r)
            mean_reward = np.mean(rewards[-50:])
            best_mean_rewards.append(mean_reward)
            if len(best_mean_rewards) > 2:
                epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
            else:
                epsilon = 0
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            iteration += 1
            r = 0
            # Epsilon decay
            model.ee_rate -= model.ee_rate * 0.01
            if model.ee_rate <= 0.2:
                model.ee_rate = 0.2
        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        # trajectories.append((prev_state, action, reward, state_value, done))
        trajectories.append((prev_state, action, state_value, reward, done))  # this is how my algorithm impl. expects data
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = model.run(state_value, reward)
        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print(model.q_table)
    print('Epsilon=', model.ee_rate)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()
    return model, trajectories, rewards

def offline_q_learning(model, training_dataset, validation_dataset, max_epochs=12, batch_size=32, gamma=0.9):
    epoch = 0
    threshold = 1e-2
    val_epoch_losses = []
    train_epoch_losses = []
    prev_val_loss = curr_val_loss = 1e10
    while threshold < curr_val_loss <= prev_val_loss and epoch < max_epochs:
        prev_val_loss = curr_val_loss
        train_losses, val_losses = model.replay(training_dataset, batch_size, validation_dataset, gamma, online=False)
        curr_val_loss = val_losses.mean()
        print('epoch {}: avg. train loss = {} & avg. val loss = {}'
              .format(epoch, train_losses.mean(), val_losses.mean()))
        train_epoch_losses.append(train_losses.mean())
        val_epoch_losses.append(val_losses.mean())
        epoch += 1
    return model, train_epoch_losses, val_epoch_losses

if __name__ == "__main__":
    SAVE = False
    val_loss_df = None
    train_loss_df = None
    online_evaluation_df = None
    # print('Start at seed {} and end before seed {}'.format(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])))
    # for SEED in range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])):
    for SEED in [2]:

        print('Using seed {}'.format(SEED))
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        env = gym.make('MountainCar-v0')
        env.seed(SEED)
        env.action_space.seed(SEED)

        # Number of states
        n_state = env.observation_space.shape[0]
        # Number of actions
        n_action = env.action_space.n
        # Number of episodes
        MAX_NUM_EPISODES = 100  # was 160
        # Learning rate
        lr = 0.001
        # Number of epochs allowed
        EPOCHS = 100

        seed_df = None
        # dataset = dataset[:1000]
        # for num_of_train_episodes in range(10, 251, 10):
        for num_of_train_episodes in [60]:
            print('num of training episodes available: {}'.format(num_of_train_episodes))
            # split train and test episodes
            # train_episodes, val_episodes = train_test_split(dataset, test_size=0.2)
            # train_episodes = train_episodes[:num_of_train_episodes]

            from rl.testbeds.mountain_car import MountainCar

            # mountain_car = MountainCar(0, 10, None, verbose=True)
            # mountain_car.play(False)
            # 10 episodes also works, but some interactions will still require ~2000 time-steps
            _, trajectories, _ = train_env(max_eps=10)
            _, val_trajectories, _ = train_env(max_eps=10)
            # _, trajectories, _ = random_play_mountain_car(max_eps=100)
            #
            # env, fis = get_fis_env()
            # print('Observation shape:', env.observation_space.shape)
            # print('Action length:', env.action_space.n)
            # action_set_length = env.action_space.n

            # model = FQLModel(gamma=0.99,
            #                  alpha=0.1,
            #                  ee_rate=1.,
            #                  action_set_length=action_set_length,
            #                  fis=fis)
            #
            # mountain_car = MountainCar(0, 100, Agent(model), verbose=True)
            # mountain_car.play(True)

            # clip_params = {'alpha': 0.1, 'beta': 0.7}
            # fis_params = {'inference_engine': 'product'}
            # # note this alpha for CQL is different from CLIP's alpha
            # cql_params = {
            #     'gamma': 0.99, 'alpha': 0.5, 'batch_size': 128, 'batches': 25,
            #     'learning_rate': 5e-2, 'iterations': 100, 'action_set_length': n_action
            # }
            # cfql = CFQLModel(clip_params, fis_params, cql_params)
            # new_cfql = CFQLModel(clip_params, fis_params, cql_params)
            X = [trajectories[0][0]]
            for idx, trajectory in enumerate(trajectories):
                X.append(trajectory[2])

            train_X = np.array(X)
            # cfql.fit(train_X, trajectories, ecm=True, Dthr=0.01, verbose=True)
            # mountain_car = MountainCar(0, 100, Agent(cfql), verbose=True)
            # # mountain_car.play(False)
            # _, _, _, greedy_offline_rewards = play_mountain_car(cfql, 101, False)
            # print(np.mean(greedy_offline_rewards[1:]))

            from sklearn.preprocessing import Normalizer

            # transformer = Normalizer().fit(train_X)
            transformer = None
            # train_X = transformer.transform(train_X)

            # get replay results
            from neuro_q_net import MIMO_replay

            rules_, weights_, antecedents_, consequents_ = unsupervised(train_X, None, ecm=True, Dthr=1e-2)
            print('There are {} rules'.format(len(rules_)))

            for input_variable in antecedents_:
                print(len(input_variable))
            # mimo = MIMO_replay(antecedents_, rules_, 2, consequents_, 0., .1)

            # print('online training FQL')
            # episodes, memory, online_train_scores = q_learning(env, mimo, MAX_NUM_EPISODES, gamma=.9, epsilon=1.0,
            #                     replay=True, title='Mamdani Neuro-Fuzzy Q-Network')
            # # online_train_fql_df = pd.DataFrame({'policy': ['online_train_FQL'] * len(online_train_scores),
            # #                                     'episode': range(len(online_train_scores)),
            # #                                     'total_reward': online_train_scores})
            # # online_train_fql_df['rules'] = len(mimo.flcs[0].rules)
            # # online_train_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes
            #
            # print('online testing FQL')
            # _, _, online_test_scores = play_cart_pole(env, mimo, 100)
            # print(np.mean(online_test_scores))
            # online_test_fql_df = pd.DataFrame({'policy': ['online_test_FQL'] * len(online_test_scores),
            #                                     'episode': range(len(online_test_scores)),
            #                                     'total_reward': online_test_scores})
            # online_test_fql_df['rules'] = len(mimo.flcs[0].rules)
            # online_test_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes

            from neuro_q_net import DoubleMIMO

            print('offline FCQL')
            # t = 0.1
            # # percent_of_data = num_of_train_episodes / len(dataset)
            # val = (num_of_train_episodes / 10) * np.log(2 + np.sqrt(3)*t)
            # cql_alpha = 1 / (1 + np.exp(val))
            cql_alpha = 0.5
            print('CQL Alpha: {}'.format(cql_alpha))  # cql alpha 0.5 with batch size 32 and 100 episodes worked well (i.e., 487.95 +- 30.50225401507239)
            offline_mimo = MIMO_replay(transformer, antecedents_, rules_, n_action, consequents_, cql_alpha=cql_alpha,
                                       learning_rate=5e-2)

            batch_size = 128
            offline_mimo, train_epoch_losses, val_epoch_losses = offline_q_learning(offline_mimo, trajectories,
                                                                                    val_trajectories, EPOCHS, batch_size,
                                                                                    gamma=0.99)  # gamma was 0.5

            from neuro_q_net import EvaluationWrapper

            avg_score, std_score = evaluate_on_environment(env)(EvaluationWrapper(offline_mimo))
            # save the training losses
            loss_df = pd.DataFrame({'policy': ['FCQL'] * len(train_epoch_losses),
                                    'epoch': range(len(train_epoch_losses)),
                                    'train_loss': train_epoch_losses, 'val_loss': val_epoch_losses})
            loss_df['train_size'] = num_of_train_episodes
            loss_df['transitions'] = len(trajectories)
            loss_df['rules'] = len(offline_mimo.flcs[0].rules)
            for idx, input_variable in enumerate(antecedents_):
                loss_df['input_variable_{}'.format(idx)] = len(input_variable)
            loss_df['avg_score'] = avg_score
            loss_df['std_score'] = std_score
            loss_df['seed'] = SEED
            print(loss_df.head())

            if seed_df is None:
                seed_df = loss_df
            else:
                seed_df = pd.concat([seed_df, loss_df])

        seed_df.to_csv('./results/mc_flc_output_{}.csv'.format(SEED), encoding='utf-8-sig', index=False)
