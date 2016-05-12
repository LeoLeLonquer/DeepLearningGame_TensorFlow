import os
from glob import glob
import tensorflow as tf

class Model(object):
	"""Abstract object representing a Reader model."""
	def __init__(self):
		self.data = None

	
