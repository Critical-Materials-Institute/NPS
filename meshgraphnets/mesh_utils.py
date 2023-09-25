
"""Utility functions for manipulating meshes."""

import scipy.spatial
import numpy as np
import time

def _DelaunayTriangulationFaces(p):
    return scipy.spatial.Delaunay(p).simplices

def get_mesh_pyfunc(points, dim=2, periodic=False):
    """ get mesh wrapped in a pyfunc."""

#   partial_fn = functools.partial(
#       DelaunayTriangulation if dim==2 else get_tetrahedron, periodic=periodic)
    faces = tf.py_function(
      _DelaunayTriangulationFaces if not periodic else (if dim==2 Periodic_2_triangulation_hierarchy_2 else  Periodic_3_Delaunay_triangulation_3),
      # https://doc.cgal.org/latest/Periodic_2_triangulation_2/index.html
      [points],
      [tf.int32])
#   faces.set_shape([None])
    return faces

# points=np.unique(np.round(np.random.rand(100,2,2)), axis=0)
points = np.array(np.meshgrid([0,1],[0,1], [0,1])).reshape((3,8)).T
print(f'debug', points, scipy.spatial.Delaunay(points).simplices)



# nbatch=10
# for NP in ():#, 4000, 10000, 40000):
#     points = np.random.rand(nbatch,NP,2)
#     corner = np.tile(np.array([[[0,0],[0,1],[1,0],[1,1]]],
#                         dtype='float32'), (nbatch,1,1))
#     points = np.concatenate([points, corner], axis=1)

#     tic = time.perf_counter()
#     tri = [scipy.spatial.Delaunay(p).simplices for p in points]
#     toc = time.perf_counter()
#     print(f"scipy NP {NP} time in {(toc - tic)/nbatch:0.4f} seconds", len(tri), tri[0].shape, tri[0])
# #    tri = torch.from_numpy(np.concatenate(tri, axis=0))#.view(1000, -1, 3)

#     from matplotlib import tri as mtri
#     import matplotlib.pyplot as plt

#     tic = time.perf_counter()
#     tri = [mtri.Triangulation(p[:, 0], p[:, 1]) for p in points]
#     toc = time.perf_counter()
#     print(f"matplotlib qhull NP {NP} time in {(toc - tic)/nbatch:0.4f} seconds", len(tri), tri[0].__class__, tri[0])
#     # ax.tripcolor(tri, velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
#     plt.triplot(tri[0], 'ko-', ms=0.5, lw=0.3)
#     plt.show()


