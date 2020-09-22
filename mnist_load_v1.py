from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

def import_graph_def():
	with open('trained_graph.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
	return

def show():
	print('=' * 60)
	for op in tf.get_default_graph().get_operations():
		print(op.name)
		for output in op.outputs:
			print('  ', output.name)
	print('=' * 60)
	return

def test():
	with tf.Session() as sess:
		print('accuracy = ', sess.run('accuracy:0', feed_dict={'x:0': mnist.test.images, 'y_:0': mnist.test.labels}))
	return

def main():
	graph = tf.Graph()
	with graph.as_default():
		import_graph_def()
		show()
		test()
	return

if __name__ == '__main__':
	main()
	