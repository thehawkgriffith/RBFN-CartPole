from agent import Agent
from run import run
import sys
import os
from datetime import datetime
import gym
from transformer import FeatureTransformer
from gym import wrappers
import numpy as np

env = gym.make('CartPole-v0')
#env._max_episode_steps = 4000
ft = FeatureTransformer(env)
agent = Agent(env, ft, "constant")
gamma = 0.99

if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


N = 300
totalrewards = np.empty(N)
for n in range(N):
    eps = 0.1*(0.97**n)
    if n == 199:
        print("Epsilon: ", eps)
    totalreward = run(env, agent, eps, gamma)
    totalrewards[n] = totalreward
    if (n + 1) % 100 == 0:
        print("Episode: ", n, "Total Reward: ", totalreward)
print("Average Reward for the last 100 Episodes: ", totalrewards[-100:].mean())
print("Total Steps: ", -totalrewards.sum())