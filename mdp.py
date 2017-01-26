#!/usr/bin/env python
import numpy as np
class MDP:
	def __init__(self, transitions, rewards):
		self.transitions = transitions
		num_actions = np.shape(transitions)[1]
		self.rewards = rewards
class MDP-R
	def __init(self, transitions):
		self.transitions = transitions
