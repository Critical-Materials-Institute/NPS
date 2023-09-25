# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2021 Fei Zhou
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
"""Model to compute gradient, hessian, etc. NO parameters"""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf
from NPS_common.tf_utils import get_activation


EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GNGradientBlock(snt.AbstractModule):
  """Compute gradient explicitly. Node feature = [displ_vec.., norm]"""

  def __init__(self, model_fn, name='GNGradientBlock'):
    super().__init__(name=name)
    # self._model_fn = model_fn

  def _update_edge_features(self, node_features, edge_set, edge_set0):
    """Aggregrates node features, and applies edge function."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    num_vert = tf.shape(sender_features)[0]
    # compute <gradient, direction_of_displ_vec>
    return tf.reshape((-sender_features+receiver_features)[:,:,None]*edge_set0.features[:,None,:-1],(num_vert,-1))/(edge_set0.features[:,-1:]**2)
    # features = [sender_features, receiver_features, edge_set.features]
    # with tf.variable_scope(edge_set.name+'_edge_fn'):
    #   return self._model_fn()(tf.concat(features, axis=-1))

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = []
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    with tf.variable_scope('node_fn'):
      return tf.concat(features, axis=-1)

  def _build(self, graph, edge_sets0):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for i,edge_set in enumerate(graph.edge_sets):
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set, edge_sets0[i])
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    # new_node_features += graph.node_features
    # new_edge_sets = [es._replace(features=es.features + old_es.features)
    #                  for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class GNGradientNet(snt.AbstractModule):
  """GraphNet model to compute gradient, hessian."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               activation='relu',
               name='EncodeProcessDecode', **kwargs):
    super().__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self.activation = get_activation(activation)

  def _make_mlp(self, output_size, layer_norm=True):
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False, activation=self.activation)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    # """Encodes node and edge features into latent features."""
    # with tf.variable_scope('encoder'):
    #   node_latents = self._make_mlp(self._latent_size)(graph.node_features)
    #   new_edges_sets = []
    #   for edge_set in graph.edge_sets:
    #     # latent = self._make_mlp(self._latent_size)(edge_set.features)
    #     new_edges_sets.append(edge_set._replace(features=latent))
    return MultiGraph(graph.node_features[:,:1], graph.edge_sets)

  def _decoder(self, graph):
    """Decodes node features from graph."""
    num_nodes = tf.shape(graph.node_features)[0]
    dim = 2
    feat = tf.reshape(graph.node_features,(num_nodes,-1)+(dim,)*self._message_passing_steps)
    ## return trace of Hessian, i.e. Laplacian, times diffusivity
    Diffusivity = 0.1
    return tf.linalg.trace(feat)*Diffusivity
    # with tf.variable_scope('decoder'):
    #   decoder = self._make_mlp(self._output_size, layer_norm=False)
    #   return decoder(graph.node_features)

  def _build(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    # model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)
    for _ in range(self._message_passing_steps):
      latent_graph = GNGradientBlock(model_fn=None)(latent_graph, graph.edge_sets)
    return self._decoder(latent_graph)

