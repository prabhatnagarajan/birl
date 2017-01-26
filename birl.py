#!/usr/bin/env python
import numpy as np

def birl(mdpr, demos):
	print "throughput"


#probability distribution P, mdp M, step size delta, and perhaps a previous policy
def PolicyWalk(P, mdp, delta):
	# Step 1 - Pick a random reward vector
	print "Selecting Random vector"
	#TODO change how I'm generating random reward vector
	rewards = np.zeros(np.shape(mdp.transitions)[0])
	# Step 2 - Policy Iteration
	policy = policy_iteration(mdp)[0]
	# Step 3
	# Step 3a
	rewards_walk = np.zeros(np.shape(mdp.transitions)[0])
	# Step 3b
	mdp.rewards = reward_walk
	Q = policy_evaluation(mdp, policy)
	# Step 3c
	


#Policy Iteration from Sutton and Barto
#assumes discount factor of 0.99
def policy_iteration(mdp, policy=None):
	#initialization
	if policy = None
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
				delta = max(delta, abs(v - value[s]))
		#policy improvement
		policy_stable = True
		for s in range(np.shape(mdp.transitions)[0]):
			old_action = policy[s]
			action_vals = [np.sum(np.add(np.dot(mdp.transitions(s,action,:), mdp.rewards),np.dot(mdp.transitions(s,action,:), np.dot(np.full((len(value)), 0.99),value)))) for action in range(np.shape(mdp.transitions)[0])]
			policy[s] = action_vals.index(max(action_vals))
			if not old_action == policy[s]:
				policy_stable = False

	return (policy, value)

#Evaluates Q^pi(s,a, R)
def policy_evaluation(mdp, policy):
	q = np.zeros(np.shape(mdp.transitions)[0],np.shape(mdp.transitions)[1])
	delta = float('inf')
	while (delta > 0.01):
		for s in range(np.shape(mdp.transitions)[0]):
			for a in range(np.shape(mdp.transitions)[1]):
				q_val = q[s,a]
				q_sum = 0.0
				for next_state in range(np.shape(mdp.transitions)[2]):
					psas = mdp.transitions[s,a,next_state]
					reward = mdp.rewards[next_state]
					qsaprime_sum = 0.0
					for next_a in range(np.shape(mdp.transitions)[1]):
						#assumes determinism
						qsaprime_sum = qsaprime_sum + 0.99 * (policy[s] == a) * q[s,a]
					q_sum = q_sum + mdp.transitions[s,a,next_state] * (reward + qsaprime_sum)
				q[s,a] = q_sum
				delta = max(delta, abs(q_val - q[s,a]))
	return q
