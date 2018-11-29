# RBFN-CartPole
An agent designed with the Radial Basis Function Network (RBFN), to play the Cart Pole environment of the OpenAIGym. The Agent contains 2 SGD Regressors, 1 for each of the possible actions, and through Q-Learning algorithm, action values are updated with each partial_fit of the current return for the previous action.
