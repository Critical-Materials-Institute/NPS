import numpy as np
import tensorflow.compat.v1 as tf
import functools

def wrap_pad_right(input, size, dim=2):
    M1 = input
    M1 = tf.concat([M1, M1[0:size]], 0)
    if dim==1:
        return M1
    M1 = tf.concat([M1, M1[:, 0:size]], 1)
    if dim==2:
        return M1
    M1 = tf.concat([M1, M1[:,:, 0:size]], 2)
    if dim==3:
        return M1
    else:
        raise ValueError(f'ERROR can only pad 1, 2 or 3d tensor')


def int2digits(ints, bases):
    dim = len(bases)
    if dim == 2:
        # return tf.stack([ints//bases[1], tf.math.floormod(ints, bases[1])], axis=-1)
        return tf.stack([ints//bases[1], ints%bases[1]], -1)
    elif dim == 3:
        ints1 = ints//bases[2]
        # return tf.stack([ints1//bases[1], tf.math.floormod(ints1, bases[1]), tf.math.floormod(ints, bases[2])], axis=-1)
        return tf.stack([ints1//bases[1], ints1%bases[1], ints%bases[2]], -1)



# def grid_points_excl_corner(Ns):
#     return np.array([x for x in grid_points_incl_corner(Ns) if np.any(x%Ns)])


# interpolator= linear_interp_coeff((4,4,4))
# # print('debug interp',interpolator)
# print(np.dot([0,0,0,0,0,0,0,1], interpolator).reshape((5,5,5)))

# print(np.dot([0,1,0,0], linear_interp_coeff((4,4))).reshape((5,5)))

# tuple(np.arange(5))
# (1,2) + (3,4)


# np.reshape(np.stack(np.meshgrid(*[np.arange(2) for _ in (2,3,4)], indexing='ij'),-1), (1,-1, 3))

# x = np.reshape(tf.range(24) , (2,6,2)) + [100, -100]

# print(tf.constant(x))

# print(tf.convert_to_tensor(x))

# # tf.scatter_nd(tf.constant([]), tf.ones(0), (3,2))
# valid = tf.where(tf.constant([[1,2]])>20)
# print(valid, tf.ones(valid.shape[0], dtype=tf.int8), (1,2,3))
# tf.scatter_nd(valid, tf.ones(valid.shape[0], dtype=tf.int8), (4,2))

# print(tf.reshape(tf.convert_to_tensor(x), -1))
# print(tf.reshape(tf.range(24),(2,6,2)) % (2,3))

# # print(grid_points([2,3,4]))

# x = np.reshape(tf.range(12) , (3,4))
# print(tf.tensor_scatter_nd_update(x, [[1], [1]], [[-999,3,3,4], [999,-1,-1,-53]]))


# x = tf.sparse.SparseTensor([[5,3],[1,63],[0,0],[5,3]], [99, -999,33,999999], (32, 64))
# print(x.values)
# print(x.indices)

# tf.scatter_nd([[0],[2],[0],[9]], [True,True,True,True], [11])
# # x = tf.reshape(x,-1)
# # print(x.values)
# # print(x.indices)

# print(tf.ones((0,5)), tf.ones((5,1)), tf.tensordot(tf.ones((0,5)), tf.ones((5,1)), 1))


# (grid_points([2]*3)-1).reshape(-1,1,3)

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


def raf1(x):
  x2 = x*x
  return x + x2 + tf.sin(x) + tf.exp(-x2/100)


def get_activation(act_string, pytorch=False):
  if act_string == 'mish':
    return mish
  elif act_string == 'raf1':
    return raf1
  elif act_string in ['sin']:
    return eval(f'tf.math.{act_string}')
  else:
    return eval(f'tf.nn.{act_string}')

