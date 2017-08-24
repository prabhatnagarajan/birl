cdef extern from "birl.c":
	int fib(int n)

import numpy as np

def main():
	print fib(5)

def select_random_reward(mdp, int step_size, double r_max):
	rewards = np.random.uniform(-r_max, r_max,np.shape(mdp.transitions)[0])
	#move theese random rewards to a gridpoint
	for i in range(len(rewards)):
		mod = rewards[i] % step_size
		if mod > (step_size/2):
			rewards[i] = rewards[i] + (step_size - mod)
		else:
			rewards[i] = rewards[i] - mod
	return rewards