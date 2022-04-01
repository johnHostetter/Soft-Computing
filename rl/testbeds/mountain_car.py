import gym
import numpy as np
import matplotlib.pyplot as plt


class MountainCar:
    def __init__(self, seed, max_episodes, agent=None,
                 verbose=False, render=False):  # None agent is random play
        self.env = gym.make("MountainCar-v0")
        self.env = self.env.unwrapped
        self.env.seed(seed)
        self.max_episodes = max_episodes
        self.agent = agent
        self.verbose = verbose
        self.render = render

        if self.verbose:
            print('Observation shape:', self.env.observation_space.shape)
            print('Action length:', self.env.action_space.n)

    def get_agent_action(self, state, initial=False):
        if self.agent is None:
            return self.env.action_space.sample()
        else:
            if initial:
                return self.agent.get_initial_action(state)
            else:
                return self.agent.get_action(state)

    def play(self, agent_is_training):  # agent_is_training is a boolean, True changes agent's parameters
        rewards = []
        r = 0
        done = True
        iteration = 0
        trajectories = []
        visited_states = []
        best_mean_rewards = []
        while iteration < self.max_episodes:
            if done:
                state_value = self.env.reset()
                visited_states.append(state_value)
                action = self.get_agent_action(state_value, initial=True)
                rewards.append(r)
                mean_reward = np.mean(rewards[-50:])
                best_mean_rewards.append(mean_reward)

                if len(best_mean_rewards) > 2:
                    epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
                else:
                    epsilon = 0

                if self.verbose:
                    try:
                        print('EPS=', iteration, ' reward=', r,
                              ' epsilon=', self.agent.model.ee_rate, ' best mean eps=', epsilon)
                    except AttributeError:
                        print('EPS=', iteration, ' reward=', r,
                              ' epsilon=', 0.0, ' best mean eps=', epsilon)

                iteration += 1
                r = 0

                if self.agent is not None:
                    # epsilon decay
                    try:
                        self.agent.model.ee_rate -= self.agent.model.ee_rate * 0.01

                        if self.agent.model.ee_rate <= 0.2:
                            self.agent.model.ee_rate = 0.2
                    except AttributeError:
                        # no ee rate given, assuming greedy policy
                        self.agent.model.ee_rate = 0.0

            # render the environment for the last couple episodes
            if self.render and iteration + 1 > (self.max_episodes - 6):
                self.env.render()

            prev_state = state_value
            state_value, reward, done, _ = self.env.step(action)
            trajectories.append((prev_state, action, reward, state_value, done))

            # change the rewards to -1
            if reward == 0:
                reward = -1

            if agent_is_training:
                action = self.agent.learn(state_value, reward)
            else:
                action = self.get_agent_action(state_value)

            r += reward

            # reach 2000 steps --> done
            if r <= -2000:
                done = True

        if self.verbose:
            try:
                print('Epsilon=', self.agent.model.ee_rate)
            except AttributeError:
                print('Epsilon=', 0.0)

        if self.render:
            plt.figure(figsize=(14, 5))
            plt.plot(best_mean_rewards[1:])
            plt.ylabel('Rewards')
            plt.show()

        self.env.close()

        return trajectories, rewards
