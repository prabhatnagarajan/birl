#!/usr/bin/env python
import numpy as np
from pdb import set_trace
class MDP:
	def __init__(self, transitions, rewards):
		self.transitions = transitions
		num_actions = np.shape(transitions)[1]
		self.rewards = rewards


	'''Policy Iteration from Sutton and Barto
	   assumes discount factor of 0.99
	   Deterministic policy iteration
	'''
	def policy_iteration(self, policy=None):
		print "Enter Policy Iteration"
		#initialization
		if policy is None:
			policy = np.zeros(np.shape(self.transitions)[0],dtype=np.int8)

		policy_stable = False
		while not policy_stable:
			#policy evaluation
			V = self.policy_evaluation(policy)
			#policy improvement
			policy_stable = True
			for state in range(np.shape(self.transitions)[0]):
				old_action = policy[state]
				action_vals = np.dot(self.transitions[state,:,:], self.rewards + 0.99 * V).tolist()
				policy[state] = action_vals.index(max(action_vals))
				if not old_action == policy[s]:
					policy_stable = False
		print "policy is"
		print policy
		print "returning"
		return (policy, value)

	'''
	policy - deterministic policy, maps state to action
	-Deterministic policy evaluation
	'''
	def policy_evaluation (self, policy):
		print "Policy Evaluation"
		V = np.zeros(np.shape(self.transitions)[0])
		delta = 1
		while delta > 0.01:
			delta = 0
			for state in range(len(V)):
				value = V[state]
				V[state] = np.dot(self.transitions[state,:,:], self.rewards + 0.99 * V)[policy[state]]
				delta = max(delta, abs(value - V[state]))
		return V

class MDPR:
	def __init__(self, transitions):
		self.transitions = transitions
