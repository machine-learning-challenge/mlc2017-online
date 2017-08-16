# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to export a model for batch prediction."""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

_TOP_PREDICTIONS_IN_OUTPUT = 2

class ModelExporter(object):

  def __init__(self, model, reader):
    self.model = model
    self.reader = reader

    with tf.Graph().as_default() as graph:
      self.inputs, self.outputs = self.build_inputs_and_outputs()
      self.graph = graph
      self.saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

  def export_model(self, model_dir, global_step_val, last_checkpoint):
    """Exports the model so that it can used for batch predictions."""

    with self.graph.as_default():
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        self.saver.restore(session, last_checkpoint)

        signature = signature_def_utils.build_signature_def(
            inputs=self.inputs,
            outputs=self.outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

        model_builder = saved_model_builder.SavedModelBuilder(model_dir)
        model_builder.add_meta_graph_and_variables(session,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save()

  def build_inputs_and_outputs(self):
    serialized_examples = tf.placeholder(tf.string, shape=(None,))

    image_id_output, index_output, predictions_output = (
        self.build_prediction_graph(serialized_examples))

    inputs = {"example_bytes":
              saved_model_utils.build_tensor_info(serialized_examples)}

    outputs = {
        "image_id": saved_model_utils.build_tensor_info(image_id_output),
        "class_indexes": saved_model_utils.build_tensor_info(index_output),
        "predictions": saved_model_utils.build_tensor_info(predictions_output)}

    return inputs, outputs

  def build_prediction_graph(self, serialized_examples):
    image_id, model_input_raw, labels_batch = (
        self.reader.prepare_serialized_examples(serialized_examples))

    model_input = model_input_raw

    with tf.variable_scope("tower"):
      result = self.model.create_model(
          model_input,
          num_classes=self.reader.num_classes,
          labels=labels_batch,
          is_training=False)

      for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)

      predictions = result["predictions"]

      prediction, index = tf.nn.top_k(predictions, 1)
    return image_id, prediction, index
