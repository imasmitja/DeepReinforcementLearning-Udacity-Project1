# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 08:04:33 2021

This is the Project number 1 from: Deep Reinforcement Learning Nanodegree Program (UDACITY)
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.


@author: Ivan Masmitja
"""

import torch
import numpy as np
from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
is_ipython = 'inline' in plt.get_backend()
if not is_ipython:
    from IPython import display
from unityagents import UnityEnvironment

def dqn(agent,file,n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=300.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), file+'_weights.pth')
            break
    return scores

#%%
"""
1. Start the Environment
The environment is already saved in the Workspace and can be accessed at the file path provided below. Please run the next code cell without making any changes.
Environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, 
and set it as the default brain we will be controlling from Python.
"""

# please do not modify the line below
env = UnityEnvironment(file_name="Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#%%
"""
2. Examine the State and Action Spaces¶
Run the code cell below to print some information about the environment.
"""
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

#%%
"""
3. Train the Agent with DQN
Run the code cell below to train the agent from scratch. You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
"""
save_file = 'DNN_Banana'
agent = Agent(state_size=state_size, action_size=action_size, seed=0, ddqn=True, per=True, dueling=True)
scores = dqn(agent, save_file)


#%%
"""
4. Plot and save the scores obtained
"""
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

import pandas as pd
def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure()
    plt.plot(scores); plt.title("Banana.exe");
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    plt.show()
    return rolling_mean

#Visualitzation:
print("[TEST] Completed {} episodes with avg. score = {}".format(len(scores), np.mean(scores)))
rolling_mean = plot_scores(scores)

#Save scores
np.savetxt(save_file+'_scores.txt', scores, delimiter = ',')

#%%

"""
5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
fig = plt.figure()
for i in range(1):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()



























