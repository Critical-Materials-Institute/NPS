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
"""Plots a CFD trajectory rollout."""

import pickle

# from absl import app
# from absl import flags
import matplotlib
from matplotlib import animation
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import numpy as np
import itertools

# FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')
# flags.DEFINE_integer('skip', 1, 'skip timesteps between animation')
# flags.DEFINE_boolean('mirrory', False, 'Make mirror plots to better see periodic boundary condition along y')
# flags.DEFINE_integer('ichannel', 0, 'which channel to show')
# flags.DEFINE_boolean('label', True, 'show label')
# flags.DEFINE_boolean('mesh_tri', False, 'show mesh of triangulation')
# flags.DEFINE_boolean('mesh', False, 'show mesh, i.e. the edges')
# flags.DEFINE_boolean('colorbar', False, 'show colorbar')
# flags.DEFINE_boolean('tri', False, 'use specified triangulation (--notri: calculate Delaunay triangulation) Do NOT set unless trigular mesh')
# flags.DEFINE_float('scale', 1.0, 'image scale WRT default')
# flags.DEFINE_string('cmap', 'viridis', 'color map')
# flags.DEFINE_string('o', '', 'save as gif')
# flags.DEFINE_boolean('debug', False, 'print sum of field')
### convert to argparse:
### sed "s/DEFINE_/ /;s/integer/int/;s/boolean/bool/;s/string/str/; s/('/ '--/" |awk 'BEGIN {FPAT = "([^ ]+)|(\"[^\"]+\")|('"'"'[^'"'"']+'"'"')"} {printf("parser.add_argument(%s type=%s, default=%s help=%s\n", $4,$3,$5,$6)}' 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('rollout_path', type=str, help='Path to rollout pickle file')
parser.add_argument('--skip', '--tskip', type=int, default=1, help='skip timesteps between animation')
parser.add_argument('--mirrory', action='store_true', default=False, help='Make mirror plots to better see periodic boundary condition along y')
parser.add_argument('--ichannel', type=int, default=0, help='which channel to show')
parser.add_argument('--label', type=bool, default=True, help='show label')
parser.add_argument('--mesh_tri', action='store_true', default=False, help='show mesh of triangulation')
parser.add_argument('--mesh', action='store_true', default=False, help='show mesh, i.e. the edges')
parser.add_argument('--colorbar', action='store_true', default=False, help='show colorbar')
parser.add_argument('--tri', action='store_true', default=False, help='use specified triangulation (--notri: calculate Delaunay triangulation) Do NOT set unless trigular mesh')
parser.add_argument('--scale', type=float, default=1.0, help='image scale WRT default')
parser.add_argument('--cmap', type=str, default='viridis', help='color map')
parser.add_argument('--o', type=str, default='', help='save as gif')
parser.add_argument('--debug', action='store_true', default=False, help='print sum of field')
FLAGS = parser.parse_args()

def run_animation(anim, fig):
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

def vector_short_pbc(vectors, cell_size):
  return vectors - np.around(vectors/cell_size)*cell_size

def main():
  if FLAGS.o: matplotlib.use('Agg')
  # matplotlib.rc('image', cmap=plt.get_cmap(FLAGS.cmap))
  plt.rcParams['image.cmap'] = FLAGS.cmap
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)
    has_GT = isinstance(rollout_data[0]['gt_velocity'][0], np.ndarray)

  if FLAGS.mirrory:
    fig, axs = plt.subplots(2, 2, figsize=(16*FLAGS.scale, 16*FLAGS.scale))
  else:
    fig, axs = plt.subplots(1, 2, figsize=(16*FLAGS.scale, 8*FLAGS.scale))
  axs = axs.flatten()
  plt.subplots_adjust(0,0,0.95, 0.95, -0.08, -0.08)
  skip = FLAGS.skip
  num_steps = len(rollout_data[0]['pred_velocity'])
  num_frames_per_rollout = (num_steps-1) // skip + 1
  num_frames = len(rollout_data) * num_frames_per_rollout
  cell_size = np.max(rollout_data[0]['mesh_pos'][0],0) - np.min(rollout_data[0]['mesh_pos'][0],0) + 1
  cutoff = 0.6*np.max(cell_size)

  # compute bounds
  bounds = []
  for trajectory in rollout_data:
    # bb_min = trajectory['gt_velocity'].min(axis=(0, 1))
    # bb_max = trajectory['gt_velocity'].max(axis=(0, 1))
    bb_min = np.concatenate(trajectory['gt_velocity' if has_GT else 'pred_velocity']).min(axis=(0))
    bb_max = np.concatenate(trajectory['gt_velocity' if has_GT else 'pred_velocity']).max(axis=(0))
    bounds.append((bb_min, bb_max))
  # print(f'debug num_steps {num_steps} num_frames_per_rollout {num_frames_per_rollout} num_frames {num_frames}  cell_size {cell_size} cutoff {cutoff} bounds {bounds}')

  def remove_boundary_face(faces, pos, cutoff):
    edges = itertools.combinations(range(len(faces[0])), 2)
    face_diameter = np.stack([np.linalg.norm(pos[faces[:,ij[0]]]-pos[faces[:,ij[1]]],axis=1) for ij in edges], -1)
    face_diameter = np.max(face_diameter, axis=-1)
    return faces[np.where(face_diameter<cutoff)]

  def animate(num):
    step = (num%num_frames_per_rollout)*skip
    traj = num//num_frames_per_rollout
    if FLAGS.debug: print(f'debug total of traj {traj} step {step} :', end='')
    for i, ax in enumerate(axs):
      col = i%2
      if (not has_GT) and (col==0): continue
      ax.cla()
      ax.set_aspect('equal')
      ax.set_axis_off()
      vmin, vmax = bounds[traj]
      pos = rollout_data[traj]['mesh_pos'][step]
      if FLAGS.tri:
        faces = rollout_data[traj]['faces'][step]
        faces = remove_boundary_face(faces, pos, cutoff)
      else:
        faces = None
      velocity = rollout_data[traj][['gt_velocity','pred_velocity'][col]][step]
      triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
      f_plot = ax.tripcolor(triang, velocity[:, FLAGS.ichannel], vmin=vmin[FLAGS.ichannel], vmax=vmax[FLAGS.ichannel])
      if FLAGS.mesh_tri: ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
      if FLAGS.label:
        ax.set_title('%s traj %d step %d' % (['GT','PD'][col], traj, step))
      if FLAGS.colorbar:
          ax.colorbar(f_plot)
      if FLAGS.mesh and (col>0):
        edges = rollout_data[traj]['faces'][step]
        lines = pos[edges]
        pbc_flag = np.where(np.linalg.norm(lines[:,0]-lines[:,1],axis=1)< cutoff)
        lines = lines[pbc_flag]
        ax.add_collection(matplotlib.collections.LineCollection(lines, colors='r', linewidths=0.7))
      if FLAGS.debug: print(f' {np.sum(velocity[:, FLAGS.ichannel])}', end='')
    if FLAGS.debug: print()
    return fig,

  anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
  run_animation(anim, fig)
  if FLAGS.o:
      anim.save(FLAGS.o, writer='imagemagick', fps=6)
  else:
      plt.show(block=True)


if __name__ == '__main__':
  main()
