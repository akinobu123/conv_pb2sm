
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
from tensorflow.python.framework import graph_util

def train_and_save():
	x = tf.placeholder(tf.float32, [None, 784], name='x')
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

	y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		max_steps = 1000
		for step in range(max_steps):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			if (step % 100) == 0:
				print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
		print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

		minimal_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y', 'accuracy'])
		tf.train.write_graph(minimal_graph, './', 'trained_graph.pb',  as_text=False)
		tf.train.write_graph(minimal_graph, './', 'trained_graph.txt', as_text=True)
	return

def main():
	graph = tf.Graph()
	with graph.as_default():
		train_and_save()
	return

if __name__ == '__main__':
	main()
