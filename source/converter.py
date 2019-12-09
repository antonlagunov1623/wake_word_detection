from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
from keras.models import load_model
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(False)

keras_model_path = '/home/anton/ITS_Partner_Lab/its_partnet_lab/1_1.h5'
wkdir = '/home/anton/ITS_Partner_Lab/its_partnet_lab/'
pb_filename = '1_1.pb'

model = load_model(keras_model_path)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)