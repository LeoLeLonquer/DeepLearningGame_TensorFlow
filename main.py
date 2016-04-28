import os
import pprint 
import tensorflow as tf

from model import GameModel

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if len(sys.argv) != 3:
		print >> sys.stderr, "usage: %s <server-name> <server-port>" % sys.argv[0]
		print >> sys.stderr, "\n"
		sys.exit(1)
		
def main(_):
	pp.pprint(flags.FLAGS.__flags)

	if not os.path.exists(FLAGS.checkpoint_dir):
	os.makedirs(FLAGS.checkpoint_dir)

	with tf.Session() as sess:
	model = GameModel(sess,sys.argv[1],sys.argv[2])

	if ! FLAGS.is_train:
	  model.load(FLAGS.checkpoint_dir)

	model.play(learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)

if __name__ == '__main__':
  tf.app.run()
