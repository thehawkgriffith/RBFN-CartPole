from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import numpy as np


class FeatureTransformer():

	'''
	FeatureTransformer class:
	Arguments:- 
	  env = Environment
	  n_components = Number of components each RBFSampler will contain
	  samples = Amount of training samples to generate
	'''

	def __init__(self, env, n_components = 500, samples = 10000):
		train_states = np.random.random((20000, 4))*2 - 2
		scaler = StandardScaler()
		scaler.fit(train_states)
		featurizer = FeatureUnion([("rbf1", RBFSampler(0.05, n_components)),
			                       ("rbf2", RBFSampler(1.0, n_components)),
			                       ("rbf3", RBFSampler(0.5, n_components)),
			                       ("rbf4", RBFSampler(0.1, n_components))])
		train_features = featurizer.fit_transform(scaler.transform(train_states))
		self.dimension = train_features.shape[1]
		self.featurizer = featurizer
		self.scaler = scaler

	def transform(self, state):
		scaled_state = self.scaler.transform(state)
		feat_state = self.featurizer.transform(scaled_state) 
		return feat_state


