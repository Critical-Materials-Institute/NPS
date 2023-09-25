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

import os; TEST_TIMING_ONLY=bool(os.environ.get('TEST_TIMING_ONLY',''))

def _rollout(model, initial_state, num_steps):
  """Rolls out a model trajectory."""
  node_type = initial_state['node_type'][:, 0]
  mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
                       tf.equal(node_type, NodeType.OUTFLOW))

  def step_fn(step, velocity, trajectory):
    prediction = model({**initial_state,
                        'velocity': velocity})
    # don't update boundary nodes
    next_velocity = tf.where(mask, prediction, velocity)
    if TEST_TIMING_ONLY:
      trajectory = trajectory.write(step, tf.reshape(tf.timestamp(),[1])+0.0*tf.cast(velocity[0,0],tf.float64))
    else:
      trajectory = trajectory.write(step, velocity)
    return step+1, next_velocity, trajectory

  _, _, output = tf.while_loop(
      cond=lambda step, cur, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['velocity'],
                 tf.TensorArray(tf.float32 if not TEST_TIMING_ONLY else tf.float64, num_steps)),
      back_prop=False, swap_memory=True,
      parallel_iterations=1)
  return output.stack()


def evaluate(model, inputs, num_steps=None, **kwargs):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  predict = True
  if num_steps is None:
    num_steps = inputs['cells'].shape[0]
    predict = False
  prediction = _rollout(model, initial_state, num_steps)

  if predict:
    scalars = {}
    traj_ops = {
        'faces': tf.repeat(inputs['cells'][0:1] if not TEST_TIMING_ONLY else tf.shape(inputs['cells'][0])[0], num_steps, axis=0),
        'mesh_pos': tf.repeat(inputs['mesh_pos'][0:1] if not TEST_TIMING_ONLY else tf.shape(inputs['mesh_pos'][0])[0], num_steps, axis=0),
        'gt_velocity': tf.constant([0.]),
        'pred_velocity': prediction
    }
  else:
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
