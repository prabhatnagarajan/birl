#!/usr/bin/env python
import numpy as np
from copy import deepcopy
import random

def birl(mdpr, demos, iterations, step_size, r_max):
	


#probability distribution P, mdp M, step size delta, and perhaps a previous policy
def PolicyWalk(mdp, step_size, iterations, r_max):
	# Step 1 - Pick a random reward vector
	mdp.rewards = select_random_reward(mdp, step_size, r_max)
	# Step 2 - Policy Iteration
	policy = mdp.policy_iteration(mdp)[0]
	# Step 3
	for i in range(iterations):	
		proposed_mdp = deepcopy(mdp)
		# Step 3a - Pick a reward vector uniformly at random from the neighbors of R
		mcmc_step(proposed_mdp, step_size, r_max)
		#Step 3b - Compute Q for policy under new reward
		Q = policy_evaluation(proposed_mdp, policy)
		# Step 3c
		#if policy is suboptimal then proceed to 3ci, 3cii, 3ciii
		if suboptimal(policy, Q):
			#3ci, do policy iteration under proposed reward function
			proposed_policy = proposed_mdp.policy_iteration(policy=policy)
			#TODO change fraction
			fraction = 0.5
			if (random.random() < min(1, fraction)):
				mdp.rewards = proposed_mdp.rewards
				policy = proposed_policy
		else:
			#TODO: Change fraction
			fraction = 0.5
			if (random.random() < min(1, fraction)):
				mdp.rewards = proposed_mdp.rewards

def mcmc_step(mdp, step_size, r_max):
	 index = random.randint(0, np.shape(mdp.transitions)[0] - 1)
	 direction = pow(-1, random.randint(0, 1))
	 '''
	 move reward at index either +step_size or -step_size, if reward
	 is too large, move it to r_max, and if it too small, move to -_rmax
	 '''
	 mdp.rewards[index] = max(min(mdp.rewards[index] + (direction * step_size), r_max),-r_max)

def suboptimal(policy, Q):
	#for every state
	for s in range(np.shape(Q)[0]):
		for a in range(np.shape(Q)[1]):
			if (Q[s, policy[s]] < Q[s,a]):
				return True
	return False

#Generates a random reward vector in the grid of reward vectors
def select_random_reward(mdp, step_size, r_max):
	rewards = np.random.uniform(-r_max, r_max,np.shape(mdp.transitions)[0])
	#move theese random rewards to a gridpoint
	for i in range(len(rewards)):
		mod = rewards[i] % step_size
		if mod > (step_size/2):
			rewards[i] = rewards[i] + (step_size - mod)
		else:
			rewards[i] = rewards[i] - mod
	return rewards

#TODO REMOVE from this file
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
