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
"""Model for neural phase simulation."""

import sonnet as snt
import tensorflow.compat.v1 as tf

from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization


def vector_pbc(vec, lattice, inv_lattice):
  """ Convert an input vector to shortest image within periodic boundary condition """
  return vec - tf.linalg.matmul(tf.math.round(tf.linalg.matmul(vec, inv_lattice)), lattice)

class Model(snt.AbstractModule):
  """Model for fluid simulation."""

  def __init__(self, learned_model, name='NPS', dim=2, periodic=False, nfeat_in=2, nfeat_out=2,
      unique_op=True, evolve=False):
    super(Model, self).__init__(name=name)
    self.dim=dim; self.periodic=bool(periodic); self.nfeat_in=nfeat_in; self.nfeat_out=nfeat_out
    self.unique_op = unique_op
    self.evolve = evolve
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = normalization.Normalizer(
          size=nfeat_out, name='output_normalizer')
      self._node_normalizer = normalization.Normalizer(
          size=nfeat_in+common.NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=dim+1, name='edge_normalizer')

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([inputs['velocity'], node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'], unique_op=self.unique_op)
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    # displacement vector under periodic boundary condition
    if self.periodic:
      # inputs['lattice']=tf.Print(inputs['lattice'], [tf.shape(inputs['lattice'])], message="debug The lat shapes are:")
      relative_mesh_pos = vector_pbc(relative_mesh_pos, inputs['lattice'], inputs['inv_lattice'])
      # relative_mesh_pos = relative_mesh_pos - tf.math.round(relative_mesh_pos/inputs['lattice'])*inputs['lattice']
    edge_features = tf.concat([
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def _build(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def featurize(self, inputs):
    graph = self._build_graph(inputs, is_training=False)
    return self._learned_model.featurize(graph)

  # @snt.reuse_variables
  def step(self, inputs, latent_graph):
    # return self._learned_model.step(latent_graph)
    latent_new, per_node_network_output = self._learned_model.step(latent_graph)
    return latent_new, self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on velocity."""
    graph = self._build_graph(inputs, is_training=True)

    # build target velocity change
    cur_velocity = inputs['velocity']
    target_velocity = inputs['target|velocity']
    target_velocity_change = target_velocity - cur_velocity
    target_normalized = self._output_normalizer(target_velocity_change)

    # build loss
    node_type = inputs['node_type'][:, 0]
    loss_mask = tf.logical_or(tf.equal(node_type, common.NodeType.NORMAL),
                              tf.equal(node_type, common.NodeType.OUTFLOW))
    if self.evolve:
      z0 = self._learned_model.featurize(graph)
      # add noise to latent features
      z0 = self._learned_model.add_latent_noise(z0, 0.01)
      z0_1 = self._learned_model.evolve(z0)
      # noise = tf.random.normal(tf.shape(inputs['target|velocity']), stddev=0.01, dtype=tf.float32)
      inputs1 = {k:(v if k != 'velocity' else inputs['target|velocity']) for k,v in inputs.items()}
      graph1 = self._build_graph(inputs1, is_training=False)
      z1 = self._learned_model.featurize(graph1)
      error_node = tf.reduce_mean((z1.node_features - z0_1.node_features)**2, axis=1)
      loss_node = tf.reduce_mean(error_node[loss_mask])
      loss_edge = 0
      for i in range(len(graph.edge_sets)):
        loss_edge += tf.reduce_mean((z1.edge_sets[i].features - z0_1.edge_sets[i].features)**2)
      loss_ae = loss_node + loss_edge
      network_output = self._learned_model.decoder(z0_1)
    else:
      network_output = self._learned_model(graph)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    if self.evolve:
      loss += loss_ae
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    velocity_update = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_velocity = inputs['velocity']
    return cur_velocity + velocity_update
