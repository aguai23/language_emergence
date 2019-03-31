import tensorflow as tf
class Actor(object):

	def __init__(self, actor_name, vocab_size=4, vocab_embedding=100,
		attribute_space=4, hidden_size=256):

		self.attribute_space = attribute_space
		self.actor_name = actor_name

		self.style = tf.placeholder(tf.float32, shape=[None, attribute_space])
		self.color = tf.placeholder(tf.float32, shape=[None, attribute_space])
		self.shape = tf.placeholder(tf.float32, shape=[None, attribute_space])

		self.hidden_size = hidden_size
		self.attribute_embeddings = self.attribute_embedding()

	def attribute_embedding(self):
		with tf.name_scope(self.actor_name):
			attribute = tf.concat([self.style, self.color, self.shape], axis=-1)
			embeddings = tf.layers.dense(attribute, hidden_size, activation=None, name="attribute_embedding", reuse=tf.AUTO_REUSE)
			return embeddings

	def listen(self, tokens):

