#!/usr/bin/env python
import numpy as np
class MDP:
	def __init__(self, transitions, rewards):
		self.transitions = transitions
		num_actions = np.shape(transitions)[1]
		self.rewards = rewards


	#Policy Iteration from Sutton and Barto
	#assumes discount factor of 0.99
	def policy_iteration(self, policy=None):
		#initialization
		if policy is None:
			policy = np.zeros(np.shape(self.transitions)[0])
		value = np.zeros(np.shape(self.transitions)[0])

		policy_stable = False
		while not policy_stable:
			#policy evaluation
			delta = float('inf')
			while delta > 0.01:
				delta = 0
				#for each s in S
				for s in range(np.shape(self.transitions)[0]):
					#v = V(s)
					v = value[s]
					action = policy[s]
					#V(s) = sum_s',r P(s',r|s pi(s))[r + YV(s')]
					value[s] = np.sum(np.dot(np.dot(mdp.transitions[s,action,:], np.add(self.rewards, np.dot(np.full(len(value),0.99), value)))))
					#value[s] = np.add(,np.dot(self.transitions(s,action,:), np.dot(np.full((len(value)), 0.99),value)))
					delta = max(delta, abs(v - value[s]))
			#policy improvement
			policy_stable = True
			for s in range(np.shape(self.transitions)[0]):
				old_action = policy[s]
				action_vals = [np.sum(np.add(np.dot(self.transitions[s,action,:], self.rewards),np.dot(self.transitions[s,action,:], np.dot(np.full((len(value)), 0.99),value)))) for action in range(np.shape(self.transitions)[0])]
				policy[s] = action_vals.index(max(action_vals))
				if not old_action == policy[s]:
					policy_stable = False

		return (policy, value)

	#Evaluates Q^pi(s,a, R)
	def policy_evaluation(self, policy):
		q = np.zeros(np.shape(self.transitions)[0],np.shape(self.transitions)[1])
		delta = float('inf')
		while (delta > 0.01):
			for s in range(np.shape(self.transitions)[0]):
				for a in range(np.shape(self.transitions)[1]):
					q_val = q[s,a]
					q_sum = 0.0
					for next_state in range(np.shape(self.transitions)[2]):
						psas = self.transitions[s,a,next_state]
						reward = self.rewards[next_state]
						qsaprime_sum = 0.0
						for next_a in range(np.shape(self.transitions)[1]):
							#assumes determinism
							qsaprime_sum = qsaprime_sum + 0.99 * (policy[s] == a) * q[s,a]
						q_sum = q_sum + self.transitions[s,a,next_state] * (reward + qsaprime_sum)
					q[s,a] = q_sum
					delta = max(delta, abs(q_val - q[s,a]))
		return q
	#MDP\R
class MDPR:
	def __init__(self, transitions):
		self.transitions = transitions
