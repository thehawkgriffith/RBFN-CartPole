#from sklearn.linear_model import SGDRegressor
import numpy as np

class SGDRegressor():

	def __init__(self, D):
		self.w = np.random.randn(D) / np.sqrt(D)
		self.lr = 10e-2

	def partial_fit(self, X, y):
		self.w += self.lr*(y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)


class Agent():

	def __init__(self, env, feature_transformer, learning_rate):
		self.env = env
		self.feature_transformer = feature_transformer
		self.action_models = []
		for _ in range(env.action_space.n):
			action_model = SGDRegressor(self.feature_transformer.dimension)
			action_model.partial_fit(self.feature_transformer.transform([self.env.reset()]), [0])
			self.action_models.append(action_model)

	def predict(self, state):
		feat_state = self.feature_transformer.transform(np.atleast_2d(state))
		Q = np.stack([m.predict(feat_state)[0] for m in self.action_models]).T
		return Q

	def update(self, state, action, Qvalue):
		feat_state = self.feature_transformer.transform(np.atleast_2d(state))
		self.action_models[action].partial_fit(feat_state, [Qvalue])

	def sample_action(self, state, epsilon):
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(state))




