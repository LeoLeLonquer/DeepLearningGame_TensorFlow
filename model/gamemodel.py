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

from .base import Model
from ops import *
from ops_network import *

debug = tools.debug
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
ACTIONS = 7 # number of valid actions
OBSERVE = 50. # timesteps to observe before training
EXPLORE = 20. # frames over which to anneal epsilon

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
		
		tf.initialize_all_variables().run()
		
		sess.run(self.step.assign(0))
		
		if load :
			self.load(checkpoint_dir)
		
		start_time = time.time()

		nb = 0
		epsilon = INITIAL_EPSILON
		
		while nb < OBSERVE:
			nb+=1
			piece_ids = situation.player_pieces.keys()
			for piece_id in piece_ids:
				self.play_random_action(piece_id)

		while nb>= OBSERVE:
			# scale down epsilon
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
			
			#train_step.run(feed_dict = {
			      
			#  })
			
			nb+=1
			
			sess.run(update) #step += 1
			
			self.communication.wait()

			debug("nb cities: %d" % len(situation.player_cities))
			debug("nb pieces: %d" % len(situation.player_pieces))

			situation.check()
			
			chunks = situation.split(8)
			
			## Define pieces in chunks
			for i in range(len(chunks)):
				chunk = chunks[i]
				debug("chunk : \n" %chunk)
				debug("chunk len: %d" %len(chunk))
				for q in range(len(chunk)):
					for r in range(len(chunk[q])):
						if chunk[q][r].visible == False:
							chunk[q][r] = 0
						else:
							if chunk[q][r].content == None:
								if chunk[q][r].terrain == ssituation.GROUND:
									chunk[q][r] = 1
								else:
									chunk[q][r] = 2
							else:
								if isinstance(chunk[q][r].content, ssituation.City):
									chunk[q][r] = 3
								elif isinstance(chunk[q][r].content, ssituation.OwnedCity):
									chunk[q][r] = 4 + chunk[q][r].content.owner # 4 et 5 !!!
								else:
									chunk[q][r] = 6 + chunk[q][r].content.piece_type + chunk[q][r].content.owner * len(situation.pieces_types)
					
					## launch Deep Q Network
					
					
					
					## Store output vector
					
			# Define cities production (TODO : define what to do with ?)
			
			for city_id in situation.player_cities:
				think.choose_relevant_random_production(situation, self.communication, city_id)
			
			# Define pieces action
			
			piece_ids = situation.player_pieces.keys()
			for piece_id in piece_ids:
				## Get the output vector and multiply it to ProbaVector
					# for ex: Water = 0 - City = 0.8 - T1 = 0.3 etc..
				
				
				## With proba Epsilon send the action ahead to the server else send a random action
				
				
				
				## Observe the action and evaluate the result (Q function)
				
				
				
				
				## Old
				self.play_random_action(piece_id)
				
			# Show situation 
			if nb % 10 == 0:
				print nb
				#situation.show()
			# Save checkpoint each 300 steps
			if nb != 0 and nb % 100 == 0:
				self.save(checkpoint_dir, step)
			# Show current progress
			step = sess.run(self.step)
			#if nb % 100 == 1:
			#	print("Epoch: [%2d] time: %4.4f, loss: %.8f" % (step, time.time() - start_time, cost))
			self.communication.end_turn()