"""
The following code was written by Seyed Saeid Masoumzadeh (GitHub user ID: seyedsaeidmasoumzadeh),
and was published for public use on GitHub under the MIT License at the following link:
    https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning 
"""

import copy
import numpy as np
import random
from empirical_fuzzy_set import EmpiricalFuzzySet as EFS
from environment import Environment
import matplotlib.pyplot as plt
import fuzzy_set as FuzzySet
import state_variable as StateVariable
import fuzzy_inference_system as FIS
import fuzzy_q_learning as FQL

def plot_results(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

# Create FIS
x1 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-2.4, -2, -1, -0.5), FuzzySet.Trapeziums(-1, -0.5, 0.5 , 1), FuzzySet.Trapeziums(0.5, 1, 2, 2.4) )
x2 = StateVariable.InputStateVariable(FuzzySet.Triangles(-2.4, -0.5, 1), FuzzySet.Triangles(-0.5, 1, 2.4))
x3 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
x4 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
fis = FIS.Build(x1,x2,x3,x4)

# Create Model
angle_list = []
ACTION_SET_LENGTH = 21
model = FQL.Model(gamma=0.9, alpha=0.1, ee_rate=0.999, 
                  q_initial_value='zero', action_set_length=ACTION_SET_LENGTH, fis=fis)
env = Environment()
reward = -1
totals = []
last_iteration = 0
for iteration in range(5000):
    if iteration % 200 == 0 or reward == -1:
        if reward == -1:
            totals.append(iteration - last_iteration)
            last_iteration = iteration
            plot_results(totals)
        env.__init__()
        action = model.get_initial_action(env.state)
        reward, state_value = env.apply_action(action)
    action = model.run(state_value, reward)
    reward, state_value = env.apply_action(action)

# play with model to collect on-policy data for offline learning
offline = {}
angle_list = []
reward = -1
totals = []
env = Environment()
num_of_episodes = 100
for episode_idx in range(num_of_episodes):
    print(episode_idx)
    episode = []
    terminated = False
    iteration = 0
    while not terminated:
        if iteration == 0:
            action = model.get_initial_action(state_value)
            prev_state = copy.deepcopy(env.state)
            reward, next_state = env.apply_action(action)
        else:
            action = model.no_learn_run(state_value, reward)
            reward, state_value = env.apply_action(action)
        
        if reward != -1:
            angle_list.append(state_value[2])
            
        elif reward == -1:
            terminated = True
            
        observation = {'state':prev_state, 'action':action, 'next_state':next_state, 'reward':reward, 'done':terminated}
        episode.append(observation)
            
    if terminated:
        env.__init__()
        action = model.get_initial_action(env.state)
        reward, state_value = env.apply_action(action)
        offline[episode_idx] = episode
        print('length of episode %s' % len(episode))
        totals.append(len(episode))

x1 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-2.4, -2, -1, -0.5), FuzzySet.Trapeziums(-1, -0.5, 0.5 , 1), FuzzySet.Trapeziums(0.5, 1, 2, 2.4) )
x2 = StateVariable.InputStateVariable(FuzzySet.Triangles(-2.4,-0.5,1), FuzzySet.Triangles(-0.5,1,2.4))
x3 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
x4 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3.14159, -1.5, 0), FuzzySet.Triangles(-1.5, 0, 1.5), FuzzySet.Triangles(0, 1.5, 3.1459))
fis = FIS.Build(x1,x2,x3,x4)

# offline?

# Create Model
angle_list = []
offline_model = FQL.Model(gamma = 1.0, alpha = 0.1 , ee_rate = 0.1, q_initial_value = 'zero',
                  action_set_length = ACTION_SET_LENGTH, fis = fis)

print('Attempting to train offline...')
for _ in range(1):
    for episode_idx in range(num_of_episodes):
        print('episode idx %s..' % episode_idx)
        episode = offline[episode_idx]
        for observation_idx in range(len(episode)):
            # (state, action, reward) = episode[observation_idx]
            state = episode[observation_idx]['state']
            action = episode[observation_idx]['action']
            reward = episode[observation_idx]['reward']        
            offline_model.run_offline(state, action, reward)
        

print(offline_model.q_table)

# Create Model
angle_list = []
#model = FQL.Model(gamma = 0.9, alpha = 0.1 , ee_rate = 0.999, q_initial_value = 'random',
#                  action_set_length = ACTION_SET_LENGTH, fis = fis)
env = Environment()
reward = -1
for iteration in range (0,1000):
    if iteration % 200 == 0 or reward == -1:
        if iteration % 200 == 0 and reward == 0:
            print('controlled')
        else:
            print('failure after %s iterations' % iteration)
        env.__init__()
        action = offline_model.induce_policy(env.state, reward)
        reward, state_value = env.apply_action(action)
    action = offline_model.induce_policy(state_value, reward)
    print(action)
    reward, state_value = env.apply_action(action)
    if reward != -1:
        angle_list.append(state_value[2])

plt.figure(figsize=(14,3))
plt.plot(angle_list)
plt.xlabel('Time')
plt.ylabel('Pole Angle')
plt.show()

env = Environment()
reward = -1
totals = []
last_iteration = 0
for iteration in range(1000):
    if iteration % 200 == 0 or reward == -1:
        if reward == -1:
            totals.append(iteration - last_iteration)
            last_iteration = iteration
            plot_results(totals)
        env.__init__()
        action = offline_model.get_initial_action(env.state)
        reward, state_value = env.apply_action(action)
    action = offline_model.run(state_value, reward)
    reward, state_value = env.apply_action(action)