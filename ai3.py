#!/usr/bin/python2.7

import random
import os
import sys
import socket
import select

import algos
import parser
import communication
import situation
import tools
import think

import tensorflow as tf

debug = tools.debug

if len(sys.argv) != 3:
	print >> sys.stderr, "usage: %s <server-name> <server-port>" % sys.argv[0]
	print >> sys.stderr, "\n"
	sys.exit(1)

def init_server():
	server_name = sys.argv[1]
	server_port = int(sys.argv[2])

	# Connect to the server.
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		server.connect((server_name, server_port))
	except:
		print "unable to connect"
		sys.exit(1)

	server_fd = server.makefile()

	situation = situation.Situation()
	parser = parser.Parser(situation)
	communication = communication.Communication(parser, server, server_fd)

def init_deep_network():
	sess = tf.InteractiveSession()
	create_network()

def create_network():
	# network weights
	W_conv1 = weight_variable([8, 8, 4, 32]) #valeurs bidons
	b_conv1 = bias_variable([32])

	# input layer
	s = tf.placeholder("float", [None, 80, 80, 4]) #valeurs bidons

	# hidden layers
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)

	h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
	h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

	# readout layer
	readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def playGame():
	n = 0
	while 1:
		n = n + 1
		communication.wait()

		debug("nb cities: %d" % len(situation.player_cities))
		debug("nb pieces: %d" % len(situation.player_pieces))

		situation.check()
		# RANDOM
		for city_id in situation.player_cities:
			think.choose_relevant_random_production(situation, communication, city_id)
		piece_ids = situation.player_pieces.keys()
		for piece_id in piece_ids:
			piece = situation.player_pieces[piece_id]
			loc = situation.get_player_piece_location(piece_id)
			depth = situation.pieces_types[piece.piece_type].speed
			heuristic = situation.get_tiles_distance
			destinations, came_from = algos.breadth_first_search_all(loc, depth, neighbors, cost, heuristic, crossable)
			if len(destinations) > 0:
				destination = random.choice(destinations)
				communication.action("moves %d %d %d" % (piece_id, destination[0], destination[1]))
		if n == 10:
			situation.show()
			n = 0
		communication.end_turn()


def main():
	init()
	init_deep_network()
	playGame()

if __name__ == "__main__":
	main()
