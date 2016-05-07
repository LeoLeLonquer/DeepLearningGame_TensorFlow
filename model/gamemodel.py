import random
import os
import sys
import socket
import select
import time

from tools import algos
from tools import parser as pparser
from tools import communication as ccommunication
from tools import situation as ssituation
from tools import tools
from tools import think

import tensorflow as tf
import random
import numpy as np
from collections import deque

from .base import Model
from ops import *
from ops_network import *

debug = tools.debug
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
ACTIONS = 7 # number of valid actions
OBSERVE = 50. # timesteps to observe before training
EXPLORE = 20. # frames over which to anneal epsilon
GAMMA = 0.99 # decay rate of past observations
REPLAY_MEMORY = 9000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
NB_CHUNK = 25
class GameModel(Model):
	"""Deep Game Network."""
	def __init__(self, sess,server_name, server_port):
		self.sess = sess
		self.build_model()
		self.init_server(server_name,server_port)
		
	def init_server(self,server_name,server_port):
		server_name = server_name
		server_port = int(server_port)

		# Connect to the server.
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			server.connect((server_name, server_port))
		except:
			print "Unable to connect"
			sys.exit(1)

		server_fd = server.makefile()

		self.situation = ssituation.Situation()
		self.parser = pparser.Parser(self.situation)
		self.communication = ccommunication.Communication(self.parser, server, server_fd)
		assert(self.communication is not None)

	def build_model(self):
		# network weights
		W_conv1 = weight_variable([4, 4, 4, 16])
		b_conv1 = bias_variable([16])
		
		W_conv2 = weight_variable([3, 3, 16, 16])
		b_conv2 = bias_variable([16])
		
		W_fc2 = weight_variable([512, ACTIONS])
		b_fc2 = bias_variable([ACTIONS])
		
		W_fc1 = weight_variable([512, 512])
		b_fc1 = bias_variable([512])
		
		# input layer
		self.input_layer = tf.placeholder("float", [None, 10, 10, 4])

		# hidden layers
		h_conv1 = tf.nn.relu(conv2d(self.input_layer, W_conv1, 2) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
		
		h_conv3_flat = tf.reshape(h_conv2, [-1, 512])
		
		self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		
		# readout layer
		self.readout = tf.matmul(self.h_fc1, W_fc2) + b_fc2
		
	def play_random_action(self, piece_id):
		situation = self.situation
		piece = situation.player_pieces[piece_id]
		loc = situation.get_player_piece_location(piece_id)
		depth = situation.pieces_types[piece.piece_type].speed
		def crossable(a):
			qa, ra = a
			return situation.is_in_map((qa, ra)) and \
					situation.can_player_piece_be_on(piece_id, (qa, ra)) and \
					situation.is_tile_none((qa, ra))
		def neighbors(a):
			qa, ra = a
			result = []
			for direction in algos.directions:
				qd, rd = direction
				qb, rb = qa + qd, ra + rd
				if situation.is_in_map((qb, rb)) and situation.can_player_piece_be_on(piece_id, (qb, rb)):
					result.append( (qb, rb) )
			return result
		def cost(x, y):
			return 1
		heuristic = situation.get_tiles_distance
		destinations, came_from = algos.breadth_first_search_all(loc, depth, neighbors, cost, heuristic, crossable)
		if len(destinations) > 0:
			dest = random.choice(destinations)
			self.communication.action("moves %d %d %d" % (piece_id, dest[0], dest[1]))
	
	def play(self, learning_rate=0.001,
            checkpoint_dir="checkpoint",load=0):
		"""
		Args:
		  learning_rate: float, The learning rate [0.001]
		  checkpoint_dir: str, The path for checkpoints to be saved [checkpoint]
		"""
		situation = self.situation
		self.learning_rate = learning_rate
		self.checkpoint_dir = checkpoint_dir

		self.step = tf.Variable(0, trainable=False)
		
		sess = tf.Session()
		
		one = tf.constant(1)
		new_value = tf.add(self.step, one)
		update = tf.assign(self.step, new_value)
		
		a = tf.placeholder("float", [None, ACTIONS])
		y = tf.placeholder("float", [None])
		
		readout_action = tf.reduce_sum(tf.mul(self.readout, a), reduction_indices = 1)
		cost = tf.reduce_mean(tf.square(y - readout_action))
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)		
		
		# store the previous observations in replay memory
    		D = deque()
    
    		# get the first state by doing nothing and preprocess the image to 80x80x4
		do_nothing = np.zeros(ACTIONS)
		do_nothing[0] = 1  # TODO : check if 0 is do_nothing
	
		# initialize all variables
		tf.initialize_all_variables().run()
		
		sess.run(self.step.assign(0))
		
		# loading networks
		if load :
			self.load(checkpoint_dir)
		
		start_time = time.time()

		t = 0
		epsilon = INITIAL_EPSILON
		
		# Static allocation to be safe
		readout_t = [None]*NB_CHUNK
		s_t = [None]*NB_CHUNK
		s_t1 = [None]*NB_CHUNK
		a_t = [[None]*ACTIONS]*NB_CHUNK # vector of vector
		action_index = [None]*NB_CHUNK
		r_t = [None]*NB_CHUNK
		
		while 1:
			self.communication.wait()

			debug("nb cities: %d" % len(situation.player_cities))
			debug("nb pieces: %d" % len(situation.player_pieces))

			situation.check()
			
			chunks = situation.split(10)
			
			for i in range(len(chunks)):
				s_t[i] = chunks[i]
				print s_t[i]
				#print len(s_t[i])
				
				#evaluate with current model 
				readout_t[i] = self.readout.eval(feed_dict = {self.input_layer : [s_t[i]]})[0]
	
				a_t[i] = np.zeros([ACTIONS])
				#choose an action for every chunk
				if random.random() <= epsilon or t <= OBSERVE:
					a_t[i][random.randrange(ACTIONS)] = 1
				else:
					# TODO : instead of taking the best, apply ponderation vector
				    	action_index = np.argmax(readout_t[i]) #this gets only the best action_index
				    	a_t[i][action_index] = 1
			
			# scale down epsilon
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
		
		
			# play actions
			piece_ids = situation.player_pieces.keys()
			for piece_id in piece_ids:
				#while can go further 
				for n in range(the_situation.pieces_types[piece.piece_type].speed):
					## Get the output vector from the right chunk and multiply it to ProbaVector
						# for ex: Water = 0 - City = 0.8 - T1 = 0.3 etc..
				
					# Send the action to the server
						# EITHER play the random action
						#self.play_random_action(piece_id)			
					
						# OR go in the right direction
					directions = the_situation.directions
					if not the_situation.is_player_piece(piece_id):
						break
					loc = the_situation.get_player_piece_location(piece_id)
					result = []
					for dir in range(len(directions)):
						next_location = loc[0] + directions[dir][0], loc[1] + directions[dir][1]
						if the_situation.can_player_piece_be_on(piece_id, next_location):
							result.append(dir)
					if len(result) > 0:
						direction = random.choice(result) # choose the one gave by the output_vector x ProbaVector
						communication.action("move %d %d" % (piece_id, direction))
					
				## Observe the action and evaluate the result (Q function)
					# check only if alive				
				if not the_situation.is_player_piece(piece_id):
					r_t[i] = -1
				else:
					r_t[i] = 1
				
				self.communication.end_turn()
				situation.check()
				chunks = situation.split(10)
				# TODO : check if game ended
				terminal = 0
				
				#state result
				for i in range(len(chunks)):
					s_t1[i] = s_t[i]
				
				# store the transition in D
				D.append((s_t[i], a_t[i], r_t, s_t1, terminal))

				if t> OBSERVE:
					# sample a minibatch to train on
					minibatch = random.sample(D, BATCH)

					# get the batch variables
					s_j_batch = [d[0] for d in minibatch]
					a_batch = [d[1] for d in minibatch]
					r_batch = [d[2] for d in minibatch]
					s_j1_batch = [d[3] for d in minibatch]

					y_batch = []
					readout_j1_batch = readout.eval(feed_dict = {self.input_layer  : s_j1_batch})
					for i in range(0, len(minibatch)):
					# if terminal only equals reward
						if minibatch[i][4]:
								y_batch.append(r_batch[i])
						else:
								y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

						# perform gradient step
						train_step.run(feed_dict = {
							y : y_batch,
							a : a_batch,
							self.input_layer : s_j_batch})

			# update the old values
			for i in range(len(chunks)):
				s_t[i] = s_t1[i]
			t += 1
			
			# Show situation 
			if t % 10 == 0:
				print t
				#situation.show()
			# Save checkpoint each 300 steps
			if t != 0 and t % 100 == 0:
				self.save(checkpoint_dir, step)
			# Show current progress
			step = sess.run(self.step)
			#if t % 100 == 1:
			#	print("Epoch: [%2d] time: %4.4f, loss: %.8f" % (step, time.time() - start_time, cost))
			
