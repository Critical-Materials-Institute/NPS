import numpy as np
import scipy
import json
import os, itertools
import tensorflow.compat.v1 as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('npy', default=None, help='array')
parser.add_argument('--periodic', action='store_true', default=False, help='Periodic boundary condition')
parser.add_argument('-o', default='output', help='output tfrecord file')
parser.add_argument('--typ', default='justNN_nodup', help='type of mesh generation')
args = parser.parse_args()


def cell2graph(a, grid_len, typ, pbc=False):
    shape = a.shape[1:-1]
    dim = len(shape)
    xyz = [np.arange(i) for i in shape]
    mesh_pos = np.stack(np.meshgrid(*xyz), axis=-1).reshape((-1, dim))
    node_type = np.zeros_like(mesh_pos[:,:1], dtype='int32')
    field = a.reshape((a.shape[0], -1, a.shape[-1])).astype('float32')
    if pbc:
        corner = mesh_pos
    else:
        corner = np.stack(np.meshgrid(*[np.arange(i-1) for i in shape]), axis=-1).reshape((-1, dim))
    assert dim in (1, 2,3), f'ERROR dim should be 1, 2 or 3 but got {dim}'
    get_partitions = lambda: partitions[np.random.choice(len(partitions))]
    if dim == 2:
        partitions = np.array([[[[0,0],[0,1],[1,0]],[[1,1],[0,1],[1,0]]], [[[0,0],[0,1],[1,1]],[[0,0],[1,0],[1,1]]]])
        if typ=='square_randomtriangle':
            pass
        elif typ=='square_1-1':
            get_partitions = lambda: partitions[0]
        elif typ=='square_11':
            get_partitions = lambda: partitions[1]
        elif typ=='square_X':
            partitions = np.array([[[[0,0],[1,0]],[[1,0],[1,1]],[[1,1],[0,1]],[[0,1],[0,0]],[[0,0],[1,1]],[[0,1],[1,0]]]])
        elif typ=='square_justNN':
            partitions = np.array([[[[0,0],[1,0]],[[1,0],[1,1]],[[1,1],[0,1]],[[0,1],[0,0]]]])
        elif typ in ["justNN_nodup", 'square_justNN_noduplicate']: # can skip tf.unique in graph generation; not safe without periodic boundary condition
            partitions = np.array([[[[0,0],[1,0]],[[0,0],[0,1]]]])
        else:
            raise ValueError(f'ERROR type {typ} not recognized')
    elif dim == 1:
        partitions = np.array([[[[0],[1]]]])
    elif dim == 3:
        partitions = np.array([
            [[0,0,0],[1,0,1],[0,1,1],[0,0,1]],[[0,0,0],[1,0,1],[1,1,0],[1,0,0]],
            [[0,0,0],[0,1,1],[1,1,0],[0,1,0]],[[0,1,1],[1,0,1],[1,1,0],[1,1,1]]#,[[0,0,0],[1,0,1],[0,1,1],[1,1,0]]
          ])
        partitions = np.stack([partitions, np.abs(partitions-np.array([[[0,0,1]]]))], 0)
        if typ in ["justNN_nodup", 'cube_justNN_noduplicate']:
            partitions = np.array([[[[0,0,0],[1,0,0]],[[0,0,0],[0,1,0]],[[0,0,0],[0,0,1]]]])
    cells = np.concatenate([ij + get_partitions() for ij in corner[:,None,None,:]])
    cells = np.mod(cells, shape)
    cells = np.dot(cells, np.cumprod((1,)+shape[:-1])).astype('int32')
    mesh_pos = (mesh_pos*(np.array(grid_len)[None,:])).astype('float32')
    return cells, mesh_pos, node_type, field

def arr2tfrecord(arr, fname, typ, verbose=True, dim=-1, grid_len=-1, periodic=False):
    """
    Converts a Numpy array a tfrecord file.
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            # return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray.flatten()))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        elif dtype_ == np.int64 or dtype_ == np.int32:
            # return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray.flatten()))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
        else:  
            raise ValueError("The input should be numpy ndarray but got {}".format(ndarray.dtype))
            
    # prepare
    f_out = fname + '.tfrecord'
    arr = np.array(arr)
    nfeat = arr.shape[-1]
    trajectory_length = arr.shape[1]
    if dim == -1:
        dim = arr.ndim - 3
    assert (dim>=1) and (dim<=3), f'Invalid dimension {dim}'
    # length of the simulation cell
    if grid_len == -1:
        grid_len = [1]*dim
    if arr.ndim == dim + 3:
        pass
    elif arr.ndim == dim + 2:
        arr = arr[..., None]
    else:
        raise f'ERROR input array should be {dim+3} dimensional'

    # write meta file
    meta_dict = {'trajectory_length': trajectory_length}
    if periodic:
        meta_dict['lattice'] = (np.diag(grid_len)*np.diag(arr.shape[2:-1])).tolist()
    meta_dict["field_names"] = [
        "cells",
        "mesh_pos",
        "node_type",
        "velocity"
    ]
    meta_dict["features"] = {
        "cells": {
            "type": "static",
            "shape": [1, -1, 2 if ('justNN' in typ) else dim+1],
            "dtype": "int32"
        },
        "mesh_pos": {
            "type": "static",
            "shape": [1, -1, dim],
            "dtype": "float32"
        },
        "node_type": {
            "type": "static",
            "shape": [1, -1, 1],
            "dtype": "int32"
        },
        "velocity": {
            "type": "dynamic",
            "shape": [trajectory_length, -1, nfeat],
            "dtype": "float32"
        }
    }
    with open(fname + ".json", "w") as outfile:
        json.dump(meta_dict, outfile)

    # write tfrecord
    writer = tf.python_io.TFRecordWriter(f_out)
    for a in arr:
        cells, mesh_pos, node_type, field = cell2graph(a, grid_len, typ, pbc=periodic)
        example = tf.train.Example(features=tf.train.Features(feature={
            "cells": _dtype_feature(cells),
            "mesh_pos": _dtype_feature(mesh_pos),
            "node_type": _dtype_feature(node_type),
            "velocity": _dtype_feature(field)
        }))
               
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    if verbose:
        print(f"Writing {f_out} done!")

arr2tfrecord(np.load(args.npy), args.o, args.typ, verbose=True, grid_len=-1, periodic=args.periodic)
