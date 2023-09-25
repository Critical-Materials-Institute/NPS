# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

# import collections
# import functools
# import sonnet as snt
import tensorflow.compat.v1 as tf

# from meshgraphnets.common import NodeType
from meshgraphnets.core_model import EncodeProcessDecode

class EncodeProcessDecodeDiffusion(EncodeProcessDecode):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _decoder(self, graph):
    """Decodes edge features to get flux."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      assert len(graph.edge_sets) == 1 # only one set of edges supported for now
      for edge_set in graph.edge_sets:
        flux = decoder(edge_set.features)
        """Aggregrates flux."""
        num_nodes = tf.shape(graph.node_features)[0]
        update = tf.math.unsorted_segment_sum(flux, edge_set.receivers, num_nodes) - \
                 tf.math.unsorted_segment_sum(flux, edge_set.senders,   num_nodes)
    return update