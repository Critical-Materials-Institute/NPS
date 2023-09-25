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
"""Runs the learner/evaluator."""

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from meshgraphnets import cfd_eval
from meshgraphnets import amr_eval
from meshgraphnets import cfd_model
from meshgraphnets import NPS_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset
# import horovod.tensorflow as hvd


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'predict'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth', 'NPS'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_enum('evaluator', None, ['cfd_eval', 'cloth_eval', 'amr_eval', 'featureevolve_eval', 'amr_featureevolve_eval'], 'Select rollout method.')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('num_predict_steps', int(100), 'No. of prediction steps')
flags.DEFINE_string('core', 'core_model', 'core model')
flags.DEFINE_integer('dim', 2, 'NPS dimension')
flags.DEFINE_integer('nfeat_in', 1, 'nfeat_in')
flags.DEFINE_integer('nfeat_out', -1, 'nfeat_out')
flags.DEFINE_integer('nfeat_latent', 128, 'nfeat_latent in GNN')
flags.DEFINE_integer('n_mpassing', 15, 'num. of message passing')
flags.DEFINE_integer('nlayer_mlp', 2, 'No. of layer in MLP')
flags.DEFINE_float('noise', -1.0, 'noise magnitude')
flags.DEFINE_integer('periodic', 0, 'NPS periodic boundary condition')
flags.DEFINE_boolean('unique_op', True, help='Apply tf.unique operator in processing edges. Turn off to speed up but be sure the specified edges have no duplicates')
flags.DEFINE_integer('n_evolve', -1, help='If >0, change core model to preprocess-step')
flags.DEFINE_string('mlp_activation', 'relu', 'Activation in MLP, e.g. relu')
flags.DEFINE_integer('batch', 4, 'batch size')
flags.DEFINE_integer('keep_ckpt', -1, 'number of checkpoints to keep. -1 to default(5)')
flags.DEFINE_integer('valid_freq', 10000, 'Perform validation/checkpoint every this many steps')
flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_integer('lr_decay', 5000000, help='Learning rate decay.')
flags.DEFINE_enum('rotate', None, ['SO', 'cubic', ''], help='Data augmentation by pointgroup operation')
flags.DEFINE_boolean('cache', False, help='Cache whole dataset into memory')
flags.DEFINE_boolean('randommesh', False, help='Data augmentation by generating random points and associated mesh')
# flags.DEFINE_float('random_lower', 0.3, 'ratio of selected points: lower bound')
# flags.DEFINE_float('random_upper', 0.8, 'ratio of selected points: upper bound')
# AMR options
flags.DEFINE_integer('amr_N', 64, 'system size, i.e. how many (fine) grids totally')
flags.DEFINE_integer('amr_N1', 1, 'how many (fine) grids to bin into one, 1 to disable')
flags.DEFINE_integer('amr_buffer', 1, 'how many buffer grids (must be 0 or 1)')
flags.DEFINE_integer('amr_eval_freq', 1, 'Call AMR in eval every this many times (default 1)')
flags.DEFINE_float('amr_threshold', 1e-3, 'threshold to coarsen regions if values are close')


PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'NPS': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=NPS_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}


def learner(model, params, mesher=None):
  """Run a learner job."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  if FLAGS.cache:
    ds = ds.cache()
  if FLAGS.randommesh:
    ds = ds.map(dataset.augment_by_randommesh, periodic=FLAGS.periodic)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  ds = dataset.split(ds)
  if FLAGS.rotate:
    ds = ds.map(dataset.augment_by_rotation(FLAGS.dim, FLAGS.rotate))
  if mesher is not None:
    ds = dataset.remesh(ds, mesher, random_translate=False)
  ds = dataset.add_training_noise(ds, noise_field=params['field'],
                                    noise_scale=params['noise'] if FLAGS.noise<0 else FLAGS.noise,
                                    noise_gamma=params['gamma'])
  ds = dataset.batch_dataset(ds, FLAGS.batch)
  # inputs = tf.data.make_one_shot_iterator(ds).get_next()
  ds_iterator = tf.data.make_initializable_iterator(ds)
  inputs = ds_iterator.get_next()

  loss_op = model.loss(inputs)
  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=FLAGS.lr,
                                  global_step=global_step,
                                  decay_steps=FLAGS.lr_decay,
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  train_op = tf.cond(tf.less(global_step, 1000),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))
  valid_op, nvalid = evaluator(model, params, 'valid', None, mesher, return_valid=True)

  saver=tf.train.Saver(max_to_keep=5 if FLAGS.keep_ckpt<=0 else FLAGS.keep_ckpt)
  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      scaffold=None,save_checkpoint_steps=None,#tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=20)) if FLAGS.keep_ckpt>0 else None,
      save_checkpoint_secs=None) as sess:
    best_valid = 1.0e99

    sess.run(ds_iterator.initializer)
    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      if step % 1000 == 0:
        logging.info('Step %d: Loss %g', step, loss)
      if step % FLAGS.valid_freq == 0:
        logging.info(f'Validating {nvalid}')
        validation_err = [list(sess.run([valid_op])[0].values()) for _ in range(nvalid)]
        valid_err_np = np.mean(validation_err,0)
        logging.info(f' Step {step} validation err {valid_err_np}')
        if valid_err_np[-1] < best_valid:
          best_valid = valid_err_np[-1]
          saver.save(sess._sess._sess._sess._sess, FLAGS.checkpoint_dir+'/model.ckpt', global_step=step)
    logging.info('Training complete.')
  evaluator(model, params, 'valid', None, mesher)


def evaluator(model, params, data_name, rollout_path, mesher=None, return_valid=False, predict_steps=None):
  """Run a model rollout trajectory."""
  nvalid = 0
  for _ in tf.python_io.tf_record_iterator(f"{FLAGS.dataset_dir}/{data_name}.tfrecord"): nvalid += 1
  ds = dataset.load_dataset(FLAGS.dataset_dir, data_name)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  if return_valid:
    ds = ds.repeat(None)
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs, mesher=mesher, num_steps=predict_steps)
  if return_valid:
    return scalar_op, nvalid
  try:
    tf.train.create_global_step()
  except:
    pass

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    mse_list = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      # error = traj_data['pred_velocity'] - traj_data['gt_velocity']
      # mse_list.append((error**2).mean(axis=1))
      if predict_steps is None:
        error = [np.mean((traj_data['pred_velocity'][i] - traj_data['gt_velocity'][i])**2, axis=0) for i in range(len(traj_data['pred_velocity']))]
        mse_list.append(error)
        scalars.append(scalar_data)
    logging.info(f'Rollout trajectory {FLAGS.num_rollouts} total done')
    if predict_steps is None:
      for key in scalars[0]:
        logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
      print(f'RMSE   total {np.sqrt(np.mean(mse_list))}')
      print(f' per_channel {np.sqrt(np.mean(mse_list,axis=(0,1)))}')
      print(f'    per_step {np.sqrt(np.mean(mse_list,axis=(0,2)))}')
      print(f'    per_traj {np.sqrt(np.mean(mse_list,axis=(1,2)))}')
    if rollout_path:
      with open(rollout_path, 'wb') as fp:
        pickle.dump(trajectories, fp)
    # variables_names = [v.name for v in tf.trainable_variables()]
    # values = sess.run(tf.trainable_variables())
    # for k, v in zip(variables_names, values):
    #     print( "Variable: ", k, v.shape, v)


def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  import sys;
  with open(f"{FLAGS.checkpoint_dir}/command.txt", "a") as f:
      f.write(' '.join(sys.argv) + '\n')
  params = PARAMETERS[FLAGS.model]
  nfeat_out = params['size'] if FLAGS.nfeat_out<0 else FLAGS.nfeat_out
  if FLAGS.n_evolve > 0:
    from meshgraphnets import featureevolve
    core_model_type = featureevolve.FeatureEvolveDecode
    # params['history'] = False # need previous state
  elif FLAGS.core == 'graph_gradient_model':
    from meshgraphnets import graph_gradient_model
    core_model_type = graph_gradient_model.GNGradientNet
  elif FLAGS.core == 'diffusion':
    from meshgraphnets import diffusion_model
    core_model_type = diffusion_model.EncodeProcessDecodeDiffusion
  else:
    core_model_type = core_model.EncodeProcessDecode
  learned_model = core_model_type(
      output_size=nfeat_out,
      activation=FLAGS.mlp_activation,
      latent_size=FLAGS.nfeat_latent,
      num_layers=FLAGS.nlayer_mlp,
      evolve_steps=FLAGS.n_evolve,
      message_passing_steps=FLAGS.n_mpassing)
  if FLAGS.model in ['NPS']:
    model = params['model'].Model(learned_model, dim=FLAGS.dim, periodic=bool(FLAGS.periodic), nfeat_in=FLAGS.nfeat_in,
    nfeat_out=nfeat_out, unique_op=FLAGS.unique_op, evolve=(FLAGS.n_evolve > 0))
  else:
    model = params['model'].Model(learned_model)
  if FLAGS.amr_N1 > 1:
    from meshgraphnets import amr
    print('''************* WARNING *************
    The present AMR implementation assumes an input field on a cubic grid of size amr_N
    ordered naturally. Make sure your dataset follows this convention''')
    mesher = amr.amr_state_variables(FLAGS.dim, [FLAGS.amr_N]*FLAGS.dim,
      [FLAGS.amr_N//FLAGS.amr_N1]*FLAGS.dim,
      tf.zeros([FLAGS.amr_N**FLAGS.dim,1],dtype=tf.float32),
      refine_threshold=FLAGS.amr_threshold, buffer=FLAGS.amr_buffer, eval_freq=FLAGS.amr_eval_freq)
    params['evaluator'] = amr_eval
  else:
    mesher = None
  if FLAGS.evaluator is not None:
    if FLAGS.evaluator == 'cfd_eval':
      params['evaluator'] = cfd_eval
    elif FLAGS.evaluator == 'cloth_eval':
      params['evaluator'] = cloth_eval
    elif FLAGS.evaluator == 'amr_eval':
      params['evaluator'] = amr_eval
    elif FLAGS.evaluator == 'featureevolve_eval':
      from meshgraphnets import featureevolve_eval
      params['evaluator'] = featureevolve_eval
    elif FLAGS.evaluator == 'amr_featureevolve_eval':
      from meshgraphnets import amr_featureevolve_eval
      params['evaluator'] = amr_featureevolve_eval
    else:
      raise ValueError(f'ERROR: unknown evaluator {FLAGS.evaluator}')
  if FLAGS.mode == 'train':
    learner(model, params, mesher)
  elif FLAGS.mode == 'eval':
    evaluator(model, params, FLAGS.rollout_split, FLAGS.rollout_path, mesher)
  elif FLAGS.mode == 'predict':
    evaluator(model, params, FLAGS.rollout_split, FLAGS.rollout_path, mesher, predict_steps=FLAGS.num_predict_steps)

if __name__ == '__main__':
  app.run(main)
