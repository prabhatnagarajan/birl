cdef extern from "birl.c":
	int fib(int n)
	double logsumexp(double* nums, unsigned int size)
	double* fast_policy_eval(double* V, double* transitions, int* pi, double* rewards, int num_states, int num_actions, double gamma, double theta, double max_value)

import numpy as np
cimport cython
cimport numpy as np
from cpython cimport array
from libc.stdlib cimport malloc, free
from pdb import set_trace
from time import time
import array

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
DTYPEB = np.uint32
ctypedef np.uint32_t DTYPEB_t

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

'''Policy Iteration from Sutton and Barto
   assumes discount factor of 0.99
   Deterministic policy iteration
'''
def policy_iteration(mdp, policy=None):
	#initialization
	if policy is None:
		policy = mdp.get_random_policy()

	policy_stable = False
	count = 0
	while not policy_stable:
		#policy evaluation
		# V = fast_eval(mdp, policy, 0.0001)
		V = policy_evaluation(mdp, policy)
		count += 1
		diff_count = 0
		#policy improvement
		policy_stable = True
		for state in mdp.states:
			old_action = policy[state]
			action_vals = np.dot(mdp.transitions[state,:,:], mdp.rewards + mdp.gamma * V).tolist()
			policy[state] = action_vals.index(max(action_vals))
			if not old_action == policy[state]:
				diff_count += 1
				policy_stable = False
	print "RETURNING POLICY"
	return (policy, V)

'''
policy - deterministic policy, maps state to action
- Deterministic policy evaluation
'''
def policy_evaluation(mdp, policy, theta=0.0001):
	# if True:
	# 	return fast_eval(mdp, policy, theta)
	# print "CALLED"
	V = np.zeros(len(mdp.states))
	print V.dtype
	while True:
		delta = 0
		for state in mdp.states:
			value = V[state]
			V[state] = np.dot(mdp.transitions[state, policy[state],:], mdp.rewards + mdp.gamma * V)
			delta = max(delta, np.abs(value - V[state]))
			#If divergence and policy has value -inf, return value function early
			if V[state] > mdp.max_value:
				return V
			if V[state] < -mdp.max_value:
				return V
		if delta < theta:
			break
	return V

def fast_eval(mdp, policy, theta=0.0001):
	cdef int num_states = len(mdp.states)
	cdef int num_actions = len(mdp.actions)
	cdef double theta_param = theta
	cdef double max_value = mdp.max_value
	cdef double gamma = mdp.gamma

	cdef array.array pi = array.array('i', policy.tolist())
	cdef array.array V = array.array('d', np.zeros(len(mdp.states)).tolist())
	cdef array.array transitions = array.array('d', np.zeros(len(mdp.states) * len(mdp.states) * len(mdp.actions)).tolist())
	cdef array.array rewards = array.array('d', mdp.rewards.tolist())

	for state in mdp.states:
		pi[state] = policy[state]
		V[state] = 0
		rewards[state] = mdp.rewards[state]
		for action in mdp.actions:
			for next_state in range(len(mdp.states)):
				transitions[state * num_actions * num_states + action * num_states + next_state] = mdp.transitions[state, action, next_state]

	cdef double * res = fast_policy_eval(V.data.as_doubles, transitions.data.as_doubles, pi.data.as_ints, rewards.data.as_doubles, 
			num_states, num_actions,  gamma, theta_param, max_value)
	result = np.zeros(len(mdp.states))
	for state in mdp.states:
		result[state] = res[state]
	return result

cdef scalar_mult(double scalar, double* vector, double* dest, int size):
	cdef int i = 0
	while (i < size):
		dest[i] = scalar * vector[i]
		i = i + 1
	return dest;

cdef vector_sum(double* vector1, double* vector2, double* dest, int size):
	cdef int i = 0
	while (i < size):
		dest[i] = vector1[i] + vector2[i]
		i = i + 1

cdef fast_policy_evaluation(double * V, double * transitions, int * pi, double  * rewards, int num_states, int num_actions, double gamma, double theta, double max_value):
	thing = np.zeros(50)
	cdef int index
	cdef int state = 0
	cdef double delta = 0
	cdef double * scalarMult =  <double *> malloc(num_states * sizeof(double))
	cdef double* vectorSum = <double *> malloc(num_states * sizeof(double))
	cdef double delta = 0.0
	cdef int state = 0
	cdef double value = 0
	while(true):
		delta = 0.0
		state = 0
		while (state < num_states):
			value = V[state]
			scalarMult(gamma, V, scalarMult, num_states)
			vector_sum(rewards, scalarMult, vectorSum, num_states)
			state = state + 1
	# while(true) {
	# 	delta = 0.0;
	# 	for (state = 0; state < num_states; state++) {
	# 		double value = V[state];
	# 		scalar_mult(gamma, V, scalarMult, num_states);
	# 		vector_sum(rewards, scalarMult, vectorSum, num_states);
	# 		index = state * num_states * num_actions + pi[state] * num_states;
	# 		V[state] = dot(transitions, index, vectorSum, num_states);
	# 		delta = fmax(delta, abs(value - V[state]));
	# 		if (V[state] > max_value) {
	# 			free(scalarMult);
	# 			free(vectorSum);
	# 			return V;
	# 		}
	# 		if (V[state] < -max_value){
	# 			free(scalarMult);
	# 			free(vectorSum);
	# 			return V;
	# 		}
	# 	}
	# 	if (delta < theta) {
	# 		break;
	# 	}
	return thing
	#return V


def c_policy_eval(mdp, policy, DTYPE_t theta):
	cdef int num_states = len(mdp.states)
	cdef int num_actions = len(mdp.actions)
	cdef int i,j,k = 0
	cdef int state = 0
	cdef np.ndarray[DTYPE_t, ndim=1] V = np.zeros(num_states, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=3] transitions = mdp.transitions.astype(dtype=DTYPE)
	cdef np.ndarray[DTYPEB_t, ndim=1] pi = policy
	cdef np.ndarray[DTYPE_t, ndim=1] rewards = mdp.rewards.astype(dtype=DTYPE)
	cdef DTYPE_t gamma = mdp.gamma
	cdef DTYPE_t delta = 0
	cdef DTYPE_t max_value = mdp.max_value
	while True:
		delta = 0
		state = 0
		while state < num_states:
			value = V[state]
			V[state] = np.dot(transitions[state, pi[state]], rewards + gamma * V)
			delta = max(delta, np.abs(value - V[state]))
			#If divergence and policy has value -inf, return value function early
			if V[state] > max_value:
				return V
			if V[state] < -max_value:
				return V
			state = state + 1
		if delta < theta:
			break
	return V

def policy_q_evaluation(mdp, policy):
	cdef np.ndarray[DTYPE_t, ndim=1] V = c_policy_eval(mdp, policy, 0.0001)
	cdef np.ndarray[DTYPE_t, ndim=2] Q = np.zeros(np.shape(mdp.transitions)[0:2])
	cdef int num_states = len(mdp.states)
	cdef int num_actions = len(mdp.actions)
	cdef int state = 0
	cdef int action = 0
	cdef np.ndarray[DTYPE_t, ndim=1] rewards = mdp.rewards.astype(dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=3] transitions = mdp.transitions.astype(dtype=DTYPE)
	cdef DTYPE_t gamma = mdp.gamma
	while state < num_states:
		action = 0
		while action < num_actions:
			Q[state, action] = np.dot(transitions[state, action], rewards + gamma * V)
			action = action + 1
		state = state + 1
	return Q

#Demos comes in the form (actual reward, demo, confidence)
def compute_log_posterior(mdp, demos, Q, prior, r_max):
	log_exp_val = 0
	cdef unsigned int num_actions = len(mdp.actions)
	cdef double* normalizer = <double*> malloc(sizeof(double) * num_actions)
	#go through each demo
	for d in range(len(demos)):
		demo = demos[d][1]
		confidence = demos[d][2]
		#for each state action pair in the demo
		for sa in demo:
			# normalizer = []
			#add to the list of normalization terms
			for a in range(np.shape(mdp.transitions)[1]):
				normalizer[a] = (confidence * Q[sa[0],a])
			'''
			We take the log of the normalizer, because we take exponent in the calling function,
			which gets rid of the log, and leaves the sum of the exponents. Also, we subtract by the log
			instead of dividing because subtracting logs can be rewritten as division
			'''
			log_exp_val = log_exp_val + confidence * Q[sa[0], sa[1]] - logsumexp(normalizer, num_actions)
	#multiply by prior
	free(normalizer)
	return log_exp_val + compute_log_prior(mdp, r_max)

def compute_log_prior(mdp, r_max):
		return (- float(len(mdp.states))) * np.log(2 * r_max)