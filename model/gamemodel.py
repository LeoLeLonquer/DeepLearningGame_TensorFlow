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
import cPickle as pickle

from .base import Model
from ops import *
from ops_network import *

debug = tools.debug
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
ACTIONS = 7 # number of valid actions
OBSERVE = 400 # timesteps to observe before training
EXPLORE = 5000 # frames over which to anneal epsilon
GAMMA = 0.99 # decay rate of past observations
REPLAY_MEMORY = 590000  # number of previous transitions to remember
MAX_SIZE_DEQUE = 10000
BATCH = 100 # size of minibatch
NB_CHUNK = 9
size = 10 #split(size)

CITY = 0
ATTACK = 1 
ELSE = 2
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
		self.parser = pparser.Parser(self.situation,[])
		self.communication = ccommunication.Communication(self.parser, server, server_fd)

	def build_model(self):
		# network weights
		W_conv1 = weight_variable([4, 4, 4, 16])
		b_conv1 = bias_variable([16])
		
		W_conv2 = weight_variable([3, 3, 16, 16])
		b_conv2 = bias_variable([16])
		
		W_fc2 = weight_variable([144, ACTIONS])
		b_fc2 = bias_variable([ACTIONS])
		
		W_fc1 = weight_variable([144, 144])
		b_fc1 = bias_variable([144])
		
		# input layer
		self.input_layer = tf.placeholder("float", [None, 10, 10, 4])

		# hidden layers
		h_conv1 = tf.nn.relu(conv2d(self.input_layer, W_conv1, 2) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
		
		h_conv3_flat = tf.reshape(h_conv2, [-1, 144])
		
		self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		
		# readout layer
		self.readout = tf.matmul(self.h_fc1, W_fc2) + b_fc2
	
	def play(self, learning_rate=0.001,
            checkpoint_dir="checkpoint",load=0):
		"""
		Args:
		  learning_rate: float, The learning rate [0.001]
		  checkpoint_dir: str, The path for checkpoints to be saved [checkpoint]
		"""
		self.learning_rate = learning_rate
		self.checkpoint_dir = checkpoint_dir

		self.step = tf.Variable(0, trainable=False)
		
		one = tf.constant(1)
		new_value = tf.add(self.step, one)
		update = tf.assign(self.step, new_value)
		
		a = tf.placeholder("float", [None, ACTIONS])
		y = tf.placeholder("float", [None])
		
		readout_action = tf.reduce_sum(tf.mul(self.readout, a), reduction_indices = 1)
		cost = tf.reduce_mean(tf.square(y - readout_action))
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)		
		
		tf.initialize_all_variables().run()
		init = tf.initialize_all_variables()
		
		# Create a summary to monitor cost function
		tf.scalar_summary("Loss", cost)

		# Merge all summaries to a single operator
		merged_summary_op = tf.merge_all_summaries()
		
		with tf.Session() as sess:
			sess.run(init)

			# Set logs writer into folder /tmp/tensorflow_logs
			summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph=sess.graph)
		
			sess.run(self.step.assign(0))
			
			# loading networks
			if load :
				self.load(checkpoint_dir)
			
			start_time = time.time()

			#GET t
			if os.path.exists('t.pckl'):
				f = open('t.pckl','rb') #read
				t = int(pickle.load(f))
			else:
				f = open('t.pckl','wb') #write
				t = 0
				pickle.dump(t, f)
			f.close()
			#Get minibatch
			if os.path.exists('d.pckl'):
				d = open('d.pckl','rb')
				D = pickle.load(d)
			else:
				d = open('d.pckl','wb')
				D = deque(maxlen=MAX_SIZE_DEQUE)
				pickle.dump(D, d)
			d.close()
			
			old_t = t
			
			epsilon = INITIAL_EPSILON - t*(INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
			if epsilon < FINAL_EPSILON:
				epsilon = FINAL_EPSILON			
			
			#Static allocation to be safe
			readout_t = [None]*NB_CHUNK
			s_t = [None]*NB_CHUNK
			s_t1 = [None]*NB_CHUNK
			a_t = [[None]*ACTIONS]*NB_CHUNK # vector of vector which contain the executed action
			r_t = [None]*NB_CHUNK
			action_done = None
			
			self.communication.wait()
			
			self.situation.check()
			player_city = self.situation.get_player_cities_number()
			chunks = self.situation.split(10)
			
			occupation = self.situation.get_occupation_player()
			for i in range(len(chunks)):
				chunk = chunks[i]
				for q in range(len(chunk)):
					for r in range(len(chunk[q])):
						if chunk[q][r].visible == False:
							chunk[q][r] = 0
						else:
							if chunk[q][r].content == None:
								if chunk[q][r].terrain == ssituation.Situation.GROUND:
									chunk[q][r] = 1
								else:
									chunk[q][r] = 2
							else:
								if isinstance(chunk[q][r].content, ssituation.City):
									chunk[q][r] = 3
								elif isinstance(chunk[q][r].content, ssituation.OwnedCity):
									chunk[q][r] = 4 + chunk[q][r].content.owner # 4 et 5 !!!
								else:
									chunk[q][r] = 6 + chunk[q][r].content.piece_type_id + chunk[q][r].content.owner * len(self.situation.piece_types)
				s_t[i] = np.stack((chunk , chunk , chunk ,chunk ), axis = 2)
			init = 0  
			ecart = 0
			last_occupation = 0
			while 1:
				if init:
					self.communication.wait()
				init = 1
				# not good if ennemy has taken a city
				last_player_city = self.situation.get_player_cities_number()
				
				#If we've lost a city
				if last_player_city - player_city < 0:
					for _i in range(len(piece_ids)):
						elem = D.pop()
						# code : D.append((s_t[i], a_t[i], r_t[i], s_t1[i], terminal))
						elem = list(elem)
						elem[2] = -10000
						D.append(elem)
				#Update for next turn
				player_city = last_player_city 
				
				#TODO : maybe select good troops ->
				for city_id in self.situation.player_cities:
					think.choose_relevant_random_production(self.situation, self.communication, city_id)
				
				# scale down epsilon
				if epsilon > FINAL_EPSILON:
					epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
				
				# play actions
				piece_ids = self.situation.player_pieces.keys()
				if t % 20 == 0:
					print("Nb piece [{}] : {} - Nb ville alies: {} / {} - Nb ville ennemie : {} - Ecart : {} - Explo : {}".format(t-old_t,len(piece_ids),self.situation.get_player_cities_number(), self.situation.get_player_cities_number()+self.situation.get_enemy_cities_number()+self.situation.get_free_cities_number(),self.situation.get_enemy_cities_number(),ecart,last_occupation))
				for piece_id in piece_ids:
					#while can go further
					piece = self.situation.player_pieces[piece_id]
					depth = self.situation.piece_types[piece.piece_type_id].speed
					#find good chunk
					i = self.situation.split_int(size,piece_id)
					readout_t[i] = self.readout.eval(feed_dict = {self.input_layer : [s_t[i]]})[0]
					ecart = np.amax(readout_t[i]) - np.amin(readout_t[i])
					while depth > 0 and self.situation.is_player_piece(piece_id):
						# check in the vector the best choice
						directions = algos.directions
						loc = piece.get_location()
						result = []
						for dir in range(len(directions)):
							next_location = loc[0] + directions[dir][0], loc[1] + directions[dir][1] # x , y
							if self.situation.can_player_piece_be_on(piece_id, next_location):
								# keep coef and check the content of the tile
								# next_location is next(x,y)
								result.append(dir)
								action_done = ELSE #at startup
								#print("Readout_t : {}".format(readout_t[i]))
								if self.situation.is_tile_free_city(next_location) or self.situation.is_tile_enemy_city(next_location):
									readout_t[i][dir]= 100000 # we take it
									action_done = CITY
								elif self.situation.is_tile_enemy_piece(next_location):
									readout_t[i][dir]=float(readout_t[i][dir])*1.5 # TODO update with kind of troops
									action_done = ATTACK
								elif self.situation.is_tile_player_piece(next_location):
									readout_t[i][dir]= - 1000000
									
							else:
								# can't play this directions
								readout_t[i][dir] = - 1000000
						
						#Bonus / Malus sur occupation de la carte
						last_occupation = self.situation.get_exploration_player()
						if last_occupation - occupation < 0:
							bonus = -10
						else:
							bonus = 10
						
						if len(result)>0:		
							if random.random() <= epsilon or t <= OBSERVE:
								direction = random.choice(result) # choose the one gave by the output_vector x ProbaVector
								#Ensure in random actions to see the correct result
								directions = algos.directions
								next_location = loc[0] + directions[direction][0], loc[1] + directions[direction][1]
								action_done = ELSE #at startup
								if self.situation.is_tile_free_city(next_location) or self.situation.is_tile_enemy_city(next_location):
									action_done = CITY
								elif self.situation.is_tile_enemy_piece(next_location):
									action_done = ATTACK
								#Play the action	
								a_t[i] = np.zeros(ACTIONS)
								a_t[i][direction] = 1
								self.communication.action("move %d %d" % (piece_id, direction))
							else:
								try:
									action_index = np.nanargmax(readout_t[i]) #this gets only the best action_index
								except ValueError:
									action_index = 6
								a_t[i] = readout_t[i]
								if action_index != 6: #action_index 6 is not moving
									self.communication.action("move %d %d" % (piece_id, action_index))
						depth = depth - 1
						
						## Observe the action and evaluate the result (Q function)
							# check only if alive				
						if not self.situation.is_player_piece(piece_id):
							if action_done == ATTACK:
								r_t[i] = -150 #not good to die during attack
							elif action_done == CITY: 
								r_t[i] = -1 #dying while taking a city is not really bad
							else:
								r_t[i] = -150 #dying by being attacked is bad 
						else:
							if action_done == ATTACK:
								r_t[i] = 150 #winning an attack
							elif action_done == CITY: 
								r_t[i] = 10000 #taking a city
							else:
								r_t[i] = 1 #being alive 
						#Bonus application
						r_t[i] += bonus
						self.situation.check()
						chunks = self.situation.split(10)
				
						for k in range(len(chunks)):
							chunk = chunks[k]	
							for q in range(len(chunk)):
								for r in range(len(chunk[q])):
									if chunk[q][r].visible == False:
										chunk[q][r] = 0
									else:
										if chunk[q][r].content == None:
											if chunk[q][r].terrain == ssituation.Situation.GROUND:
												chunk[q][r] = 1
											else:
												chunk[q][r] = 2
										else:
											if isinstance(chunk[q][r].content, ssituation.City):
												chunk[q][r] = 3
											elif isinstance(chunk[q][r].content, ssituation.OwnedCity):
												chunk[q][r] = 4 + chunk[q][r].content.owner # 4 et 5 !!!
											else:
												chunk[q][r] = 6 + chunk[q][r].content.piece_type_id + chunk[q][r].content.owner * len(self.situation.piece_types)
							chunk = np.reshape(chunk,(10,10,1))
							#print(chunk)
							s_t1[k] =  np.append(chunk, s_t[k][:,:,1:], axis = 2)
						# TODO : check if game end
						terminal = 0						
						
						# store the transition in D
						D.append((s_t[i], a_t[i], r_t[i], s_t1[i], terminal))
						if len(D) > REPLAY_MEMORY:
							D.popleft()

				if t> OBSERVE:
					# sample a minibatch to train on
					if BATCH < len(D):
						minibatch = random.sample(D, BATCH)
					else : 
						try:
							minibatch = random.sample(D,len(D))
						except ValueError:
							print("ValueError : sample population")
							minibatch = random.sample(D,1)
					# get the batch variables
					s_j_batch = [d[0] for d in minibatch]
					a_batch = [d[1] for d in minibatch]
					r_batch = [d[2] for d in minibatch]
					s_j1_batch = [d[3] for d in minibatch]
					
					y_batch = []
					readout_j1_batch = self.readout.eval(feed_dict = {self.input_layer : s_j1_batch})
					for i in range(0, len(minibatch)):
					# if terminal only equals reward
						if minibatch[i][4]:
							y_batch.append(r_batch[i])
						else:
							y_batch.append(r_batch[i] + GAMMA * float(np.max(readout_j1_batch[i])))
							
					train_step.run(feed_dict = {
						y : y_batch,
						a : a_batch,
						self.input_layer : s_j_batch})
					# Write logs at every iteration
					summary_str = sess.run(merged_summary_op, feed_dict = {
						y : y_batch,
						a : a_batch,
						self.input_layer : s_j_batch})
					summary_writer.add_summary(summary_str, t)
				# update the old values
				self.communication.end_turn()
				t += 1
				# Save checkpoint each 100 steps
				if t != 0 and t % 100 == 0:
					self.save(checkpoint_dir, step)
					f = open('t.pckl', 'wb')
					pickle.dump(t, f)
					f.close()
					d = open('d.pckl', 'wb')
					pickle.dump(D, d)
					d.close()
				# Show current progress
				step = sess.run(self.step)
				if t % 100 == 1:
					print("Epoch: [%2d], player: %d time: %4.4f, epsilon: %.8f" % (t-old_t,self.situation.player_id, time.time() - start_time, epsilon))
