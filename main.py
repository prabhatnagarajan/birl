#!/usr/bin/env python
import gridworld
from gridworld import GridWorld
from mdp import MDP
import numpy as np
from birl import *
from constants import *
from prior import *
import cbirl

def initialize_gridworld(width, height):
	# where 24 is a goal state that always transitions to a 
	# special zero-reward terminal state (25) with no available actions
	num_states = 6 * 6
	trans_mat = np.zeros((num_states, 4, num_states))

	# NOTE: the following iterations only happen for states 0-23.
	# This means terminal state 25 has zero probability to transition to any state, 
	# even itself, making it terminal, and state 24 is handled specially below.

	# Action 1 = down
	for s in range(num_states):
		if s < num_states - width:
			trans_mat[s,1,s + width] = 1
		else:
			trans_mat[s,1,s] = 1
	  
		# Action 0 = up
	for s in range(num_states):
		if s >= width:
			trans_mat[s, 0, s-width] = 1
		else:
			trans_mat[s, 0, s] = 1
	  
	# Action 2 = left
	for s in range(num_states):
		if s % width > 0:
			trans_mat[s, 2, s - 1] = 1
		else:
			trans_mat[s, 2, s] = 1
	  
	# Action 3 = right
	for s in range(num_states):
		if s%width < width - 1:
			trans_mat[s,3,s+1] = 1
		else:
			trans_mat[s,3,s] = 1

	# Finally, goal state always goes to zero reward terminal state
	for a in range(4):
		for s in range(num_states):
			trans_mat[num_states - 1, a, s] = 0
		trans_mat[num_states - 1, a, num_states - 1] = 1 

	return trans_mat

def initialize_rewards(dims, num_states):
	weights = np.random.normal(0, 0.25, dims)
	rewards = dict()
	for i in range(num_states):
		rewards[i] = np.dot(weights, np.random.normal(-3, 1, dims))
		#Give goal state higher value
	rewards[num_states - 1] = 10
	return rewards

if __name__ == '__main__':
	cbirl.main()
	exit()
	transitions = initialize_gridworld(6, 6)
	mdp = MDP(transitions, initialize_rewards(5, 36), 0.99)
	#mdp = MDP(0, range(4), [35], 0.99, range(36), initialize_rewards(5, 36), transitions)
	thing = GridWorld(mdp)
	demos = thing.record(1)
	print demos
	policy = birl(mdp, 0.02, 1000, 1.0, demos, 500, 20, PriorDistribution.UNIFORM)
	print "Finished BIRL"
	print "Agent Playing"
	reward, playout = thing.play(policy)
	print "Reward is " + str(reward)
	print "Playout is " + str(playout)
