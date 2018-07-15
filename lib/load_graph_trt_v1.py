import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import copy

import os
import yaml
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph

class LoadFrozenGraph():
    """
    LOAD FROZEN GRAPH
    TRT Graph
    """
    def __init__(self, cfg):
        self.cfg = cfg
        return

    def load_graph(self):
        print('Building Graph')
        trt_graph_def=self.build_trt_graph()
        if not self.cfg['split_model']:
            # force CPU device placement for NMS ops
            for node in trt_graph_def.node:
                if 'NonMaxSuppression' in node.name:
                    node.device = '/device:CPU:0'
                else:
                    node.device = '/device:GPU:0'
            return self.non_split_trt_graph(graph_def=trt_graph_def)
        else:
            return self.split_trt_graph(graph_def=trt_graph_def)

    def print_graph(self, graph):
        """
        PRINT GRAPH OPERATIONS
        """
        print("{:-^32}".format(" operations in graph "))
        for op in graph.get_operations():
            print(op.name,op.outputs)
        return

    def print_graph_def(self, graph_def):
        """
        PRINT GRAPHDEF NODE NAMES
        """
        print("{:-^32}".format(" nodes in graph_def "))
        for node in graph_def.node:
            print(node)
        return

    def print_graph_operation_by_name(self, graph, name):
        """
        PRINT GRAPH OPERATION DETAILS
        """
        op = graph.get_operation_by_name(name=name)
        print("{:-^32}".format(" operations in graph "))
        print("{:-^32}\n{}".format(" op ", op))
        print("{:-^32}\n{}".format(" op.name ", op.name))
        print("{:-^32}\n{}".format(" op.outputs ", op.outputs))
        print("{:-^32}\n{}".format(" op.inputs ", op.inputs))
        print("{:-^32}\n{}".format(" op.device ", op.device))
        print("{:-^32}\n{}".format(" op.graph ", op.graph))
        print("{:-^32}\n{}".format(" op.values ", op.values()))
        print("{:-^32}\n{}".format(" op.op_def ", op.op_def))
        print("{:-^32}\n{}".format(" op.colocation_groups ", op.colocation_groups))
        print("{:-^32}\n{}".format(" op.get_attr ", op.get_attr("T")))
        i = 0
        for output in op.outputs:
            op_tensor = output
            tensor_shape = op_tensor.get_shape().as_list()
            print("{:-^32}\n{}".format(" outputs["+str(i)+"] shape ", tensor_shape))
            i += 1
        return

    # helper function for split model
    def node_name(self, n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def non_split_trt_graph(self, graph_def):
        tf.import_graph_def(graph_def, name='')
        return tf.get_default_graph()

    def split_trt_graph(self, graph_def):
        """
        Load frozen_graph and split it into half of GPU and CPU.
        """
        split_shape = self.cfg['split_shape']
        num_classes = self.cfg['num_classes']

        SPLIT_TARGET_SLICE1_NAME = 'Postprocessor/Slice' # Tensor
        SPLIT_TARGET_EXPAND_NAME = 'Postprocessor/ExpandDims_1' # Tensor

        tf.reset_default_graph()

        """ ADD CPU INPUT """
        slice1_in = tf.placeholder(tf.float32, shape=(None, split_shape, num_classes), name=SPLIT_TARGET_SLICE1_NAME)
        expand_in = tf.placeholder(tf.float32, shape=(None, split_shape, 1, 4), name=SPLIT_TARGET_EXPAND_NAME) # shape=output shape

        """
        Load placeholder's graph_def.
        """
        for node in tf.get_default_graph().as_graph_def().node:
            if node.name == SPLIT_TARGET_SLICE1_NAME:
                slice1_def = node
            if node.name == SPLIT_TARGET_EXPAND_NAME:
                expand_def = node

        tf.reset_default_graph()

    
        """
        Check the connection of all nodes.
        edges[] variable has input information for all nodes.
        """
        edges = {}
        name_to_node_map = {}
        node_seq = {}
        seq = 0
        for node in graph_def.node:
            n = self.node_name(node.name)
            name_to_node_map[n] = node
            edges[n] = [self.node_name(x) for x in node.input]
            node_seq[n] = seq
            seq += 1

        """
        Alert if split target is not in the graph.
        """
        dest_nodes = [SPLIT_TARGET_SLICE1_NAME, SPLIT_TARGET_EXPAND_NAME]
        for d in dest_nodes:
            assert d in name_to_node_map, "%s is not in graph" % d

        """
        Making GPU part.
        Follow all input nodes from the split point and add it into keep_list.
        """
        nodes_to_keep = set()
        next_to_visit = dest_nodes

        while next_to_visit:
            n = next_to_visit[0]
            del next_to_visit[0]
            if n in nodes_to_keep:
                continue
            nodes_to_keep.add(n)
            next_to_visit += edges[n]

        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

        keep = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
            keep.node.extend([copy.deepcopy(name_to_node_map[n])])

        """
        Making CPU part.
        It removes GPU part from loaded graph and add new inputs.
        """
        nodes_to_remove = set()
        for n in node_seq:
            if n in nodes_to_keep_list: continue
            nodes_to_remove.add(n)
        nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

        remove = graph_pb2.GraphDef()
        remove.node.extend([slice1_def])
        remove.node.extend([expand_def])
        for n in nodes_to_remove_list:
            remove.node.extend([copy.deepcopy(name_to_node_map[n])])

        """
        Import graph_def into default graph.
        """
        with tf.device('/gpu:0'):
            tf.import_graph_def(keep, name='')
        with tf.device('/cpu:0'):
            tf.import_graph_def(remove, name='')

        #self.print_graph_operation_by_name(tf.get_default_graph(), SPLIT_TARGET_SLICE1_NAME)
        #self.print_graph_operation_by_name(tf.get_default_graph(), SPLIT_TARGET_EXPAND_NAME)

        """
        return    
        """
        return tf.get_default_graph()

    def build_trt_graph(self):
        MODEL             = self.cfg['model']
        PRECISION_MODE    = self.cfg['precision_model']
        CONFIG_FILE       = "data/" + MODEL + '.config'   # ./data/ssd_inception_v2_coco.config 
        CHECKPOINT_FILE   = 'data/' + MODEL + '/model.ckpt'    # ./data/ssd_inception_v2_coco/model.ckpt
        FROZEN_MODEL_NAME = MODEL+'_trt_' + PRECISION_MODE + '.pb'
        TRT_MODEL_DIR     = 'data'
        LOGDIR            = 'logs/' + MODEL + '_trt_' + PRECISION_MODE

        config_path, checkpoint_path = download_detection_model(MODEL, 'data')

        frozen_graph_def, input_names, output_names = build_detection_graph(
            config=CONFIG_FILE,
            checkpoint=CHECKPOINT_FILE
        )

        tf.reset_default_graph()
        trt_graph_def = trt.create_inference_graph(
            input_graph_def=frozen_graph_def,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode=PRECISION_MODE,
            minimum_segment_size=50
        )
        tf.train.write_graph(trt_graph_def, TRT_MODEL_DIR,
                             FROZEN_MODEL_NAME, as_text=False)

        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(tf.get_default_graph())
        train_writer.flush()
        train_writer.close()

        return trt_graph_def
