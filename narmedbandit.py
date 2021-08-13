import sys
import abc
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
	
class BanditAgent:
	def __init__(self, n):
		__metaclass__ = abc.ABCMeta
		self.n = n
		self.qs = np.random.normal(0,1,size=n)
		self.Q = np.zeros(n)
		self.N = np.zeros(n)
		
	@abc.abstractmethod
	def choose_action(self):
		return
			
	def update(self, action, get_reward=False):
		reward = self.qs[action] + np.random.normal(0,1)
		if self.N[action] > 0:
			self.Q[action] += (reward - self.Q[action]) / self.N[action]
		else:
			self.Q[action] = reward
		self.N[action] += 1
		if get_reward:
			return reward
	
class EpsGreedyBanditAgent(BanditAgent):
	def __init__(self, n, epsilon):
		self.epsilon = epsilon
		BanditAgent.__init__(self,n)
		
	def choose_action(self):
		if random.random() < self.epsilon:
			return np.random.choice(range(self.n))
		else:
			return np.argmax(self.Q)
			
class OptimisticEpsGreedyBanditAgent(EpsGreedyBanditAgent):
	def __init__(self, n, epsilon, optimism=5):
		self.optimism = optimism
		EpsGreedyBanditAgent.__init__(self,n,epsilon)
		self.Q = np.ones(n)*optimism
		
class UCBBanditAgent(BanditAgent):
	def __init__(self, n, c):
		self.c = c
		self.t = 1
		BanditAgent.__init__(self,n)
		
	@staticmethod
	def ucb_score(Q, N, c, t):
		return Q + c * np.sqrt((np.log(t)) / N)
		
	def choose_action(self):
		ucb_scores = []
		for i in range(len(self.Q)):
			if self.N[i] > 0:
				ucb_scores.append(UCBBanditAgent.ucb_score(self.Q[i],self.N[i],self.c,self.t))
			else:
				ucb_scores.append(float("inf"))
		action = np.argmax(ucb_scores)
		self.t += 1
		return action		
			
def test_eps_greedy(agent_class, n, epsilon, steps=1000, agent_testbed=2000):
	rewards, accuracy = np.zeros(steps), np.zeros(steps)
	cnt = 1
	for _ in tqdm(range(agent_testbed)):
		agent = agent_class(n,epsilon)
		max_q = np.argmax(agent.qs)
		for step in range(steps):
			action = agent.choose_action()
			reward = agent.update(action, get_reward=True)
			rewards[step] += (reward - rewards[step]) / cnt
			accuracy[step] += (int(action==max_q) - accuracy[step]) / cnt
		cnt += 1
	return rewards, accuracy
	
def test_ucb(agent_class, n, c, steps=1000, agent_testbed=2000):
	rewards, accuracy = np.zeros(steps), np.zeros(steps)
	cnt = 1
	for _ in tqdm(range(agent_testbed)):
		agent = agent_class(n,c)
		max_q = np.argmax(agent.qs)
		for step in range(steps):
			action = agent.choose_action()
			reward = agent.update(action, get_reward=True)
			rewards[step] += (reward - rewards[step]) / cnt
			accuracy[step] += (int(action==max_q) - accuracy[step]) / cnt
		cnt += 1
	return rewards, accuracy