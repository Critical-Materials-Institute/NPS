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
"""Functions to build evaluation metrics for CFD data."""

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType
from meshgraphnets import core_model


def _rollout(model, initial_state, num_steps):
  """Rolls out a model trajectory."""
  node_type = initial_state['node_type'][:, 0]
  mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
                       tf.equal(node_type, NodeType.OUTFLOW))
  initial_latent = model.featurize(initial_state)

  def step_fn(step, z_node, z_edges, velocity, trajectory):
    latent = core_model.MultiGraph(
        node_features=z_node,
        edge_sets= [core_model.EdgeSet(
         name=es.name,
         features=z_edges[ies],
         receivers=es.receivers,
         senders=es.senders) for ies, es in enumerate(initial_latent.edge_sets)])
    latent_new, prediction = model.step({'velocity':velocity}, latent)
    # don't update boundary nodes
    next_velocity = tf.where(mask, prediction, velocity)
    trajectory = trajectory.write(step, velocity)
    return step+1, latent_new.node_features, [es.features for es in latent_new.edge_sets], next_velocity, trajectory

  _, _, _, _, output = tf.while_loop(
      cond=lambda step, zn, ze, cur, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_latent.node_features, [es.features for es in initial_latent.edge_sets], initial_state['velocity'],
                 tf.TensorArray(tf.float32, num_steps)),
      back_prop=False, swap_memory=True,
      parallel_iterations=1)
  return output.stack()


def evaluate(model, inputs, **kwargs):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  num_steps = inputs['cells'].shape[0]
  prediction = _rollout(model, initial_state, num_steps)

  error = tf.reduce_mean((prediction - inputs['velocity'])**2, axis=-1)
  scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_velocity': inputs['velocity'],
      'pred_velocity': prediction
  }
  return scalars, traj_ops
