#!/usr/bin/env python
import numpy as np

def birl(mdpr, demos):
	print "throughput"

#probability distribution P, mdp M, step size delta
def PolicyWalk(P, mdp, delta):
	print "thing"

#Policy Iteration from Sutton and Barto
#assumes discount factor of 0.99
def policy_iteration(mdp):
	#initialization
	policy = np.zeros(np.shape(mdp.transitions)[0])
	value = np.zeros(np.shape(mdp.transitions)[0])

	policy_stable = False
	while not policy_stable:
		#policy evaluation
		delta = float('inf')
		while delta > 0.01:
			delta = 0
			#for each s in S
			for s in range(np.shape(mdp.transitions)[0]):
				#v = V(s)
				v = value[s]
				action = policy[s]
				#V(s) = sum_s',r P(s',r|s pi(s))[r + YV(s')]
				value[s] = np.sum(np.add(np.dot(mdp.transitions(s,action,:), mdp.rewards),np.dot(mdp.transitions(s,action,:), np.dot(np.full((len(value)), 0.99),value))))
				delta = max(delta, abs(v = value[s]))
		#policy improvement
		policy_stable = True
		for s in range(np.shape(mdp.transitions)[0]):
			old_action = policy[s]
			action_vals = [np.sum(np.add(np.dot(mdp.transitions(s,action,:), mdp.rewards),np.dot(mdp.transitions(s,action,:), np.dot(np.full((len(value)), 0.99),value)))) for action in range(np.shape(mdp.transitions)[0])]
			policy[s] = action_vals.index(max(action_vals))
			if not old_action == policy[s]:
				policy_stable = False

	return (policy, value)
