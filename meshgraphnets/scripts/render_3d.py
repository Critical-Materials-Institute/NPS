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
"""
rendering 3D rollout

"""  # pylint: disable=line-too-long

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}


def main(unused_argv):

  if not FLAGS.rollout_path:
    raise ValueError("A `rollout_path` must be passed.")
  with open(FLAGS.rollout_path, "rb") as file:
    rollout_data = pickle.load(file)

  # fig, axes = plt.subplots(1, 2)
  fig = plt.figure(figsize=(10, 5))
  axes = [fig.add_subplot(121+i, projection='3d') for i in range(2)]

  plot_info = []
  for ax_i, (label, rollout_field) in enumerate(
      [("Ground truth", "ground_truth_rollout"),
       ("Prediction", "predicted_rollout")]):
    # Append the initial positions to get the full trajectory.
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]], axis=0)
    ax = axes[ax_i]
    ax.set_title(label)
    bounds = rollout_data["metadata"]["bounds"]
    ax.set_xlim(0, bounds[0][0])
    ax.set_ylim(0, bounds[1][1])
    ax.set_zlim(0, bounds[2][2])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    plot_info.append((ax, trajectory, ax.scatter(*(trajectory[0].T), "o", s=9, color=[TYPE_TO_COLOR[i] for i in rollout_data["particle_types"]])))
    # ax.set_aspect(1.)
    # points = {
    #     particle_type: ax.scatter([], [], [], "o", s=9, color=color)[0]
    #     for particle_type, color in TYPE_TO_COLOR.items()}
    # plot_info.append((ax, trajectory, points))

  num_steps = trajectory.shape[0]

  def update(step_i):
    outputs = []
    for _, trajectory, sub_fig in plot_info:
      sub_fig._offsets3d = tuple(trajectory[step_i].T)
      # for particle_type, line in points.items():
      #   mask = rollout_data["particle_types"] == particle_type
      #   line.set_data(*[trajectory[step_i, mask, ix] for ix in range(2)])
      #   outputs.append(line)
    return outputs

  unused_animation = animation.FuncAnimation(
      fig, update,
      frames=np.arange(0, num_steps, FLAGS.step_stride), interval=10)
  plt.show(block=FLAGS.block_on_show)


if __name__ == "__main__":
  app.run(main)
