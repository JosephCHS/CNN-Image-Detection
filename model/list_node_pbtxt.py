import tensorflow as tf
from google.protobuf import text_format

with open('graph.pbtxt') as f:
    graph_def = text_format.Parse(f.read(), tf.GraphDef())

print ([n.name for n in graph_def.node])
