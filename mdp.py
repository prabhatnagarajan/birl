#!/usr/bin/env python
import numpy as np
from pdb import set_trace
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
			policy = np.zeros(np.shape(self.transitions)[0],dtype=np.int8)
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
					value[s] = np.sum(np.dot(self.transitions[s,action,:], np.add(self.rewards, np.dot(np.full(len(value),0.99), value))))
					#value[s] = np.add(,np.dot(self.transitions(s,action,:), np.dot(np.full((len(value)), 0.99),value)))
					delta = max(delta, abs(v - value[s]))

			#policy improvement
			policy_stable = True
			for s in range(np.shape(self.transitions)[0]):
				old_action = policy[s]
				action_vals = np.zeros(np.shape(self.transitions)[0])
				for action in range(np.shape(self.transitions)[1]):
					action_vals[action] = np.sum(np.add(np.dot(self.transitions[s,action,:], self.rewards),np.dot(self.transitions[s,action,:], np.dot(np.full((len(value)), 0.99),value))))
				policy[s] = np.argmax(action_vals)
				if not old_action == policy[s]:
					policy_stable = False
		print "policy is"
		print policy
		print "returning"
		return (policy, value)


	#Evaluates Q^pi(s,a, R)
	def policy_evaluation(self, policy):
		print "evaluating policy"
		q = np.zeros(np.shape(self.transitions)[0:2])
		delta = float('inf')
		while (delta > 0.01):
			delta = 0
			for s in range(np.shape(self.transitions)[0]):
				for a in range(np.shape(self.transitions)[1]):
					q_val = q[s,a]
					q_sum = 0.0
					for next_state in range(np.shape(self.transitions)[2]):
						psas = self.transitions[s,a,next_state]
						reward = self.rewards[next_state]
						if (np.isnan(q_sum)):
							print "NAN"
							exit()
						q_sum = q_sum + psas * (reward + 0.99 * q[next_state, policy[next_state]])
					q[s,a] = q_sum
					delta = max(delta, abs(q_val - q[s,a]))
					if np.isnan(delta):
						print "NAN"
		return q

class MDPR:
	def __init__(self, transitions):
		self.transitions = transitions
