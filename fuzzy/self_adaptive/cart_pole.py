import torch
import numpy as np

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
        callable: scorer function.


    """

    def scorer(algo, *args):
        # other = algo
        # algo = algo.agent.model
        episode_rewards = []
        rule_activations_during_end = []  # keep track of what rules led to end of episode
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    try:
                        action = torch.argmax(algo.predict([observation])[0]).item()
                    except AttributeError or TypeError:  # for 'list' object has no attribute 'shape'
                        # _ = other.predict([observation])[0]
                        # rules = other.agent.flcs[0].current_rule_activations
                        action = torch.argmax(algo.predict(torch.tensor(np.array([observation])))).item()
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if done:
                    try:
                        rule_activations_during_end.append(algo.agent.flcs[0].current_rule_activations)
                    except AttributeError:
                        rule_activations_during_end.append(0.0)
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.std(episode_rewards), rule_activations_during_end

    return scorer