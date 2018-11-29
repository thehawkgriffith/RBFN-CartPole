import numpy as np
def run(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
    	action = model.sample_action(observation, eps)
    	prev_observation = observation
    	observation, reward, done, info = env.step(action)
    	if done:
    		reward = -200
    	next = model.predict(observation)
    	G = reward + gamma*np.max(next)
    	model.update(prev_observation, action, G)
    	if reward == 1:
    		totalreward += reward
    	iters += 1
    return totalreward

