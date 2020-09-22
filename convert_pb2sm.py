import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def convert_pb_to_saved_model(pb_model_path, export_dir, in1, in2, out):
	model_graph = read_pb_model(pb_model_path)
	save_model(model_graph, export_dir, in1, in2, out)

def read_pb_model(pb_model_path):
	with tf.gfile.GFile(pb_model_path, "rb") as f:
		model_graph = tf.GraphDef()
		model_graph.ParseFromString(f.read())
		return model_graph

def save_model(model_graph, export_dir, in1, in2, out):
	builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

	sigs = {}
	with tf.Session(graph=tf.Graph()) as sess:
		tf.import_graph_def(model_graph, name="")
		g = tf.get_default_graph()
		in1_tensor = g.get_tensor_by_name(in1)
		in2_tensor = g.get_tensor_by_name(in2)
		out_tensor = g.get_tensor_by_name(out)

		sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
			tf.saved_model.signature_def_utils.predict_signature_def(
				{"input1": in1_tensor, "input2": in2_tensor}, {"output": out_tensor})

		builder.add_meta_graph_and_variables(sess,
											 [tag_constants.SERVING],
											 signature_def_map=sigs)
		builder.save()


def main():
	convert_pb_to_saved_model('trained_graph.pb', 'trained_model_v2', 'x:0', 'y_:0', 'accuracy:0')

main()
