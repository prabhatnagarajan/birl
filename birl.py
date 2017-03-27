#!/usr/bin/env python
import numpy as np
from copy import deepcopy
import random
import math
from scipy.misc import logsumexp
from constants import *
from prior import *
from pdb import set_trace
#Demos is a dictionary of demonstration mapping to alpha (confidence)
#step_size is the step size for MCMC
#r_max is the maximum possible reward in the domain
def birl(mdp, step_size, iterations, r_max, demos, burn_in, sample_freq, prior):
	if not isinstance(prior, PriorDistribution):
		print "Invalid Prior"
		raise ValueError
	reward_samples = PolicyWalk(mdp, step_size, iterations, burn_in, sample_freq, r_max, demos, prior)
	mdp.rewards = np.mean(reward_samples)
	print "Rewards are "
	print mdp.rewards
	#Optimal deterministic policy
	print "computed mean rewards"
	optimal_policy = mdp.policy_iteration()[0]
	print "Done with policy Iteration"
	return optimal_policy


#probability distribution P, mdp M, step size delta, and perhaps a previous policy
#Returns : List of Sampled Rewards
def PolicyWalk(mdp, step_size, iterations, burn_in, sample_freq, r_max, demos, prior):
	reward_samples = []
	# Step 1 - Pick a random reward vector
	mdp.rewards = select_random_reward(mdp, step_size, r_max)
	# Step 2 - Policy Iteration
	policy = mdp.policy_iteration()[0]
	#initialize an original posterior, will be useful later
	post_orig = None
	# Step 3
	for i in range(iterations):	
		proposed_mdp = deepcopy(mdp)
		# Step 3a - Pick a reward vector uniformly at random from the neighbors of R
		mcmc_step(proposed_mdp, step_size, r_max)
		#Step 3b - Compute Q for policy under new reward
		Q = proposed_mdp.policy_q_evaluation(policy)
		# Step 3c
		if post_orig is None:
			post_orig = compute_log_posterior(mdp, demos, mdp.policy_q_evaluation(policy), prior, r_max)
		#if policy is suboptimal then proceed to 3ci, 3cii, 3ciii
		if suboptimal(policy, Q):
			#3ci, do policy iteration under proposed reward function
			proposed_policy = proposed_mdp.policy_iteration(policy=policy)[0]
			'''
			Take fraction of posterior probability of proposed reward and policy over 
			posterior probability of original reward and policy
			'''
			post_new = compute_log_posterior(proposed_mdp, demos, proposed_mdp.policy_q_evaluation(proposed_policy), prior, r_max)
			fraction = np.exp(post_new - post_orig)
			if (random.random() < min(1, fraction)):
				mdp.rewards = proposed_mdp.rewards
				policy = proposed_policy
				post_orig = post_new
		else:
			'''
			Take fraction of the posterior probability of proposed reward under original policy over
			posterior probability of original reward and original policy
			'''
			post_new = compute_log_posterior(proposed_mdp, demos, Q, prior, r_max)
			fraction = np.exp(post_new - post_orig)
			if (random.random() < min(1, fraction)):
				mdp.rewards = proposed_mdp.rewards
				post_orig = post_new
		print i
		#Store samples
		if i >= burn_in:
			if i % sample_freq == 0:
				print i
				reward_samples.append(mdp.rewards)
	#Step 4 - return the reward samples
	return reward_samples

#Demos comes in the form (actual reward, demo, confidence)
def compute_log_posterior(mdp, demos, Q, prior, r_max):
	log_exp_val = 0
	#go through each demo
	for d in range(len(demos)):
		demo = demos[d][1]
		confidence = demos[d][2]
		#for each state action pair in the demo
		for sa in demo:
			normalizer = []
			#add to the list of normalization terms
			for a in range(np.shape(mdp.transitions)[1]):
				normalizer.append(confidence * Q[sa[0],a])
			'''
			We take the log of the normalizer, because we take exponent in the calling function,
			which gets rid of the log, and leaves the sum of the exponents. Also, we subtract by the log
			instead of dividing because subtracting logs can be rewritten as division
			'''
			log_exp_val = log_exp_val + confidence * Q[sa[0], sa[1]] - logsumexp(normalizer)
	#multiply by prior
	return log_exp_val + compute_log_prior(prior, mdp, r_max)

def compute_log_prior(prior, mdp, r_max):
	if prior == PriorDistribution.UNIFORM:
		return (- float(len(mdp.states))) * np.log(2 * r_max)


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