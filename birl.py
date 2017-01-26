#!/usr/bin/env python
import numpy as np

def birl(mdpr, demos):
	print "throughput"

#probability distribution P, mdp M, step size delta
def PolicyWalk(P, mdp, delta):
	print "thing"

def policy_iteration(mdp):
	#initialization
	policy = np.zeros(np.shape(mdp.transitions)[0])
	value = np.zeros(np.shape(mdp.transitions)[0])

	#policy evaluation
	delta = float('inf')
	while delta > 0.01:
		delta = 0
	#policy iteration
