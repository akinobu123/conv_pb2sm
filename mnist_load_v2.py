import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

# load saved model
def load_saved_model(input_folder_path, in_sigs, out_sigs):
	model_meta_graph = tf.saved_model.load(input_folder_path)
	model_func = model_meta_graph.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
	#	model_func = model_meta_graph.signatures["serving_default"]
	return model_func.prune(feeds=in_sigs, fetches=out_sigs)

def main():
	# load mnist data
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	test_labels = test_labels[:1000]					# take out the first 1000 datas
	test_labels = tf.one_hot(test_labels, depth=10)		# convert to one-hot type data
	test_images = tf.cast(test_images[:1000].reshape(-1, 784) / 255.0, tf.float32)

	# load saved model
	model_func = load_saved_model('trained_model_v2', ['x:0', 'y_:0'], ['accuracy:0'])
#	input_signature = model_func.inputs
#	output_signature = model_func.outputs
#	print(input_signature[0].name)
#	print(input_signature[1].name)
#	print(output_signature[0].name)

	@tf.function
	def execute_model(x, y_):
		return model_func(x, y_)

#	class Exportable(tf.Module):
#		@tf.function
#		def __call__(self, x, y_): return pruned_model_func(x, y_)
#
#	exported_model = Exportable()
#	acc = exported_model(test_images, test_labels)

	print('###### start inference ######')
	acc = execute_model(test_images, test_labels)
	print('Accuracy = ', acc[0].numpy())
	return

if __name__ == '__main__':
	main()
	