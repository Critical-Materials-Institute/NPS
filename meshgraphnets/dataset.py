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
"""Utility functions for reading the datasets."""

import functools
import json
import os

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType


def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  if 'lattice' in meta:
    import numpy as np
    latt = np.array(meta['lattice'])
    out['lattice'] = tf.tile(tf.constant(latt[None,...], dtype='float32'), [meta['trajectory_length'], 1, 1])
    out['inv_lattice'] = tf.tile(tf.constant(np.linalg.inv(latt)[None,...], dtype='float32'), [meta['trajectory_length'], 1, 1])
    # out['lattice'] = tf.tile(tf.constant(np.diag(latt)[None,None,...], dtype='float32'), [meta['trajectory_length'], 1, 1])
  return out


def load_dataset(path, split):
  """Load dataset."""
  meta_file = os.path.join(path, split+'.json')
  if not os.path.exists(meta_file):
    meta_file = os.path.join(path, 'meta.json')
  with open(meta_file, 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
  """Splits trajectories into frames, and adds training noise."""
  ds = split(ds)
  ds = add_training_noise(ds, noise_field, noise_scale, noise_gamma)
  return ds


def split(ds):
  """Splits trajectories into frames."""
  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  return ds


def add_training_noise(ds, noise_field, noise_scale, noise_gamma):
  """Adds training noise."""
  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=noise_scale, dtype=tf.float32)
    # don't apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    return frame

  ds = ds.map(add_noise, num_parallel_calls=8)
  ds = ds.shuffle(10000)
  ds = ds.repeat(None)
  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def batch_dataset(ds, batch_size):
  """Batches input datasets."""
  shapes = ds.output_shapes
  types = ds.output_types
  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key == 'cells':
        # renumber node indices in cells
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      elif key in ['lattice', 'inv_lattice']:
        out[key] = ds_val.reduce(initial, lambda prev, cur: cur)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out

  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds


def augment_by_rotation(dim, pointgroup='SO'):
  """Augment a single stack of inputs by rotation.
  Args:
    inputs
  Returns:
    randomly rotated.

  """
  # dim_mesh = inputs['mesh_pos'].shape[-1]
  # cosine and sine of three angles
  import math
  sign = 1 # tf.cast(2*tf.random.uniform(shape=[], maxval=2, dtype=tf.int32)-1, tf.float32)
  if dim == 3:
    if pointgroup == 'SO':
      cs = tf.random.normal((3,2))
      cs = tf.linalg.normalize(cs, axis=1)[0]
    elif pointgroup in ['Oh', 'cubic']:
      cs = tf.cast(tf.random.uniform(shape=[3], maxval=4, dtype=tf.int32), tf.float32)*math.pi/4
      cs = tf.stack([tf.cos(cs), tf.sin(cs)], 1) * sign
    else:
      raise ValueError(f'ERROR unknown pointgorup {pointgroup} in {dim}-D')
    rotation = tf.stack([cs[0,0]*cs[1,0], cs[0,0]*cs[1,1]*cs[2,1]-cs[0,1]*cs[2,0], cs[0,0]*cs[1,1]*cs[2,0]+cs[0,1]*cs[2,1],
                        cs[0,1]*cs[1,0], cs[0,1]*cs[1,1]*cs[2,1]+cs[0,0]*cs[2,0], cs[0,1]*cs[1,1]*cs[2,0]-cs[0,0]*cs[2,1],
                        -cs[1,1],        cs[1,0]*cs[2,1],                         cs[1,0]*cs[2,0]])
    rotation = tf.transpose(tf.reshape(rotation, (3,3)))
  elif dim == 2:
    if pointgroup == 'SO':
      cs = tf.random.normal((1,2))
      cs = tf.linalg.normalize(cs, axis=1)[0]
    elif pointgroup in ['4m', 'cubic']:
      cs = tf.cast(tf.random.uniform(shape=[], maxval=8, dtype=tf.int32), tf.float32)*math.pi/4
      cs = tf.stack([tf.cos(cs), tf.sin(cs)*sign])
    rotation = tf.stack([[cs[0], -cs[1]],
                         [cs[1],  cs[0]]])
    # rotation = tf.transpose(tf.reshape(rotation, (2,2)))
  else:
    raise ValueError(f'WARNING rotation not implemented for dimension {dim}')
  # new_latt = tf.linalg.matmul(inputs['lattice'][0], rotation)

  def _rotate(inputs):
    # print(f'debug inputs {inputs.__class__} {inputs}')
    out = {}
    for key, val in inputs.items():
      # print(f'debug k {key} v {val}')
      if key in ('mesh_pos','lattice'):
        out[key] = tf.linalg.matmul(val, rotation)
      else:
        out[key] = val
    if 'lattice' in inputs.keys():
      out['inv_lattice'] = tf.linalg.inv(out['lattice'])
    return out
  return _rotate


def augment_by_randommesh(inputs, ratio):
  """Augment a single stack of inputs by randomly selecting points and triangulation
       based on the selected points.
  Args:
    inputs
    ratio = (lower limit, higher limit)
  Returns:
    randomly rotated.

  """
  print(f'debug inputs {inputs.__class__} {inputs}')
  # n_inputs = tf.random(ratio[0], ratio[1]).cast(int32)
  # cosine and sine of three angles
  # input points in inputs['mesh_pos']: [N_sequence, N_pts, dim] float32
  #   input mesh inputs['faces']: [N_sequence, N_faces, dim+1] int32
  #   lattice inputs['lattice'] inputs['inv_lattice'] # ignored
  # 1 randomly select N_inputs points from mesh_pos
  # 2 call generate_mesh on selected points
  # change dict inputs['mesh_pos'] = new selected points
  # change dict inputs['faces'] = new triangles.
  # return 
  raise ValueError('ERROR: random mesh is NOT implemented yet')


def remesh(ds, mesher, random_translate=False):
  dim = mesher.dim
  def fn(trajectory):
    mesh = trajectory['mesh_pos']
    field_inp = tf.reshape(trajectory['velocity'], mesher.shape_all+(1,))
    field_tgt = tf.reshape(trajectory['target|velocity'], mesher.shape_all+(1,))
    if random_translate:
      shift = tf.random.uniform([dim], [0]*dim, mesher.shape1, dtype=tf.int32)
      field_inp = tf.roll(field_inp, shift, list(range(dim)))
      field_tgt = tf.roll(field_tgt, shift, list(range(dim)))
    keys = ('mesh_pos', 'node_type', 'velocity', 'target|velocity', 'cells')
    values = mesher.remesh_input(mesh, field_inp, field_tgt, input_dense=True)
    out = dict(zip(keys, values))
    for key, val in trajectory.items():
      if not key in keys:
        out[key] = val
    return out

  ds = ds.map(fn, num_parallel_calls=8)
  return ds
