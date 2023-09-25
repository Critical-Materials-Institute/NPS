# Lint as: python3
# pylint: disable=g-bad-file-header
# ============================================================================
"""Functions to build evaluation metrics for adaptive remeshing."""

import tensorflow.compat.v1 as tf
from meshgraphnets import amr

## ignoreing node type for simplified bookkeeping
# from meshgraphnets.common import NodeType

import os; TEST_TIMING_ONLY=bool(os.environ.get('TEST_TIMING_ONLY',''))

def _rollout(model, initial_state, num_steps, mesher, field_gt, return_gt=False):
  """Rolls out a model trajectory."""
  node_type = initial_state['node_type'][:, 0]
  # mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
  #                      tf.equal(node_type, NodeType.OUTFLOW))

  def step_fn(step, mesh, node_type, field, edges, refine_index, coarse_index, mesh_traj, field_traj, edge_traj, fgt_traj):

    # do_mesh = lambda: mesher.remesh_input(mesh, field, field_gt[step],
    #     refine_index=refine_index, coarse_index=coarse_index, input_dense=False, index=True)
    # skip_mesh = lambda: (mesh, node_type, field, tf.gather_nd(tf.transpose(tf.reshape(field_gt[step],mesher.shape_all))[...,None],tf.cast(mesh,tf.int32)), edges, refine_index, coarse_index)
    def do_mesh():
      return mesher.remesh_input(mesh, field, field_gt[step] if return_gt else None,
        refine_index=refine_index, coarse_index=coarse_index, input_dense=False, index=True)

    def skip_mesh():
      return (mesh, node_type, field, tf.gather_nd(tf.transpose(tf.reshape(field_gt[step],mesher.shape_all))[...,None],tf.cast(mesh,tf.int32)) if return_gt else tf.constant(0.), edges, refine_index, coarse_index)

    mesh, node_type, field, fgt, edges, refine_index, coarse_index =\
     tf.cond(tf.equal(step%mesher.eval_freq, 0), do_mesh, skip_mesh)
    if TEST_TIMING_ONLY:
      mesh_traj = mesh_traj.write(step, tf.cast(tf.convert_to_tensor(tf.shape(mesh)[0]),tf.float32))
      # field_traj = field_traj.write(step, tf.cast(tf.convert_to_tensor(tf.shape(field)),tf.float32))
      field_traj = field_traj.write(step, tf.reshape(tf.timestamp(),[1]))
      edge_traj = edge_traj.write(step, tf.convert_to_tensor(tf.shape(edges)[0]))
      # t = tf.timestamp()
      # t = t - 1000* tf.math.floordiv(t, 1000)
      # t = tf.cast(tf.reshape(t,[1]),tf.float32)
      # fgt_traj = fgt_traj.write(step, t)
      fgt_traj = fgt_traj.write(step, tf.constant([0.]))#tf.reshape(tf.timestamp(),[1]))
    else:
      mesh_traj = mesh_traj.write(step, mesh)
      field_traj = field_traj.write(step, field)
      edge_traj = edge_traj.write(step, edges)
      fgt_traj = fgt_traj.write(step, fgt)
    prediction = model({**initial_state,
        'velocity': field, 'mesh_pos': mesh, 'node_type': node_type, 'cells': edges})
    # # don't update boundary nodes
    # next_velocity = tf.where(mask, prediction, velocity)
    field_next = prediction
    return step+1, mesh, node_type, field_next, edges, refine_index, coarse_index, mesh_traj, field_traj, edge_traj, fgt_traj

  def _read_fn(x):
    return [x.read(i) for i in range(num_steps)]

  i0 = tf.constant(0)
  _, _,_,_,_,_,_, mesh_traj, field_traj, edge_traj, fgt_traj = tf.while_loop(
      cond=lambda step, *unused: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(i0, initial_state['mesh_pos'], initial_state['node_type'], initial_state['velocity'], initial_state['cells'],
                 mesher.refine_index, mesher.coarse_index,
                 tf.TensorArray(tf.float32, num_steps, infer_shape=False),
                 tf.TensorArray(tf.float32 if not TEST_TIMING_ONLY else tf.float64, num_steps, infer_shape=False),
                 tf.TensorArray(tf.int32, num_steps, infer_shape=False),
                 tf.TensorArray(tf.float32, num_steps, infer_shape=False)),
      shape_invariants=(i0.get_shape(),
        tf.TensorShape([None, mesher.dim]), tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), tf.TensorShape([None, 2]),
        tf.TensorShape([None, 1]),tf.TensorShape([None, 1]),
        tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
      back_prop=False, swap_memory=True,
      parallel_iterations=1)
  # fgt_traj = [tf.gather_nd(field_gt[i], tf.cast(mesh_traj.read(i),tf.int32)) for i in range(num_steps)]
  return _read_fn(mesh_traj), _read_fn(field_traj), _read_fn(edge_traj), _read_fn(fgt_traj)


def evaluate(model, inputs, mesher, num_steps=None):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  predict = True
  if num_steps is None:
    num_steps = inputs['cells'].shape[0]
    predict = False
  mesh_traj, field_traj, edge_traj, fgt_traj = _rollout(model, initial_state, num_steps, mesher, inputs['velocity'], return_gt=not predict)

  if predict:
    scalars = {}
  else:
    error = [tf.reduce_mean((field_traj[i] - fgt_traj[i])**2) for i in range(num_steps)]
    scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
  traj_ops = {
      'faces': edge_traj,
      'mesh_pos': mesh_traj,
      'gt_velocity': fgt_traj,
      'pred_velocity': field_traj
  }
  return scalars, traj_ops
