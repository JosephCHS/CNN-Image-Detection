import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="Freeze the trained modele to a .pb file and a .bytes files")
parser.add_argument("ID_checkpoint", metavar="ID", type=str,
                    help = "ID of the checkpoint,  model.ckpt-<ID_checkpoint>.index")
args = parser.parse_args()

saver = tf.train.import_meta_graph("./model.ckpt-" + args.ID_checkpoint + ".meta", clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./model.ckpt-" + args.ID_checkpoint)
output_node_names="action"
output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
							        input_graph_def,
							        output_node_names.split(","))
output_graph="model.pb"
output_graph_unity="model.bytes"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
with tf.gfile.GFile(output_graph_unity, "wb") as f:
    f.write(output_graph_def.SerializeToString())
sess.close()

# Other way to freeze the graph with TF tools
#  python3 -m tensorflow.python.tools.freeze_graph --input_graph=./graph.pbtxt --input_checkpoint=./model.ckpt-662 --output_graph=./frozen_graph.pb --output_node_names='save/restore_all' --input_binary=false
