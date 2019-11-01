import sys
import tensorflow as tf
import argparse
from tensorflow.python.tools import freeze_graph

MODEL_NAME = "graph"

# Parse arguments to find the ckpt ID

parser = argparse.ArgumentParsert(description="Find ID ckpt.")
parser.add_argument("id-ckpt", type=str)
args = parser.parse_args()

# Freeze the graph

input_graph_path = MODEL_NAME+".pbtxt"
checkpoint_path = "./model.ckpt-" + args.id-ckpt
input_saver_def_path = ""
input_binary = False
output_node_names = "action"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = "frozen_"+MODEL_NAME+"PAELLA"+".bytes"
clear_devices = True

freeze_graph.freeze_graph(input_graph_path,
	    		  input_saver_def_path,
                          input_binary,
			  checkpoint_path,
			  output_node_names,
                          restore_op_name,
			  filename_tensor_name,
                          output_frozen_graph_name,
			  clear_devices,
			  "")
