__author__ = 'Fei Zhou'

# Data augmentation with point group symmetry

import numpy as np


# def _tp(x):
#   return np.swapaxes(x, 3, 4)


# def _tp_ch_last(x):
#   return np.swapaxes(x, 2, 3)

def op0(x): return x
ix = 3

def op4m_1(x): return x.flip((ix, ix+1))

def op4m_2(x): return x.flip((ix+1,))

def op4m_3(x): return x.flip((ix,))

def op4m_4(x): return x.swapaxes(ix,ix+1).flip((ix+1,))

def op4m_5(x): return x.swapaxes(ix,ix+1).flip((ix,))

def op4m_6(x): return x.swapaxes(ix,ix+1).flip((ix,ix+1))

def op4m_7(x): return x.swapaxes(ix,ix+1)


class PointGroup(object):
  """ Point group symmetry
      Note the argument of the point group ops are assumed to be of the shape
      array[nbatch, nframes, nchannel, nx, ny, ...]
  """

  def __init__(self, name, dim=2, channel_first=True):
    self.channel_first = channel_first
    self.name = name
    self.dim = dim
    if self.dim == 2:
        self.init2d()
    elif self.dim == 3:
        self.init3d()
    else:
        raise "Unsupported dimension %d"%(dim)
    print(f'Point group {self.name} for dim {self.dim} initialized')

  def init2d(self):
    ix = 3 if self.channel_first else 2
    self.ops = [op0]
    if self.name in ['', '1', None]:
      self.name = '1'
    elif self.name == '2':
      self.ops+= [lambda x: x.flip((ix, ix+1))]
    elif self.name == 'mx':
      self.ops+= [lambda x: x.flip((ix+1,))]
    elif self.name == 'my':
      self.ops+= [lambda x:  x.flip((ix,))]
    elif self.name == 'mm':
      self.ops+= [lambda x: x.flip((ix, ix+1)), lambda x: x.flip((ix,)), lambda x: x.flip((ix+1,))]
    elif self.name == '4':
      self.ops+= [lambda x: x.swapaxes(ix,ix+1).flip((ix+1,)), lambda x: x.swapaxes(ix,ix+1).flip((ix,)), lambda x: x.flip((ix,ix+1))]
    elif self.name == '4m':
      # self.ops+= [lambda x: x.flip((ix, ix+1)), lambda x: x.flip((ix+1,)), lambda x: x.flip((ix,)), \
      #             lambda x: x.swapaxes(ix,ix+1).flip((ix+1,)), lambda x: x.swapaxes(ix,ix+1).flip((ix,)), lambda x: x.swapaxes(ix,ix+1).flip((ix,ix+1)), \
      #             lambda x: x.swapaxes(ix,ix+1)]
      if ix==3:
        self.ops+= [op4m_1, op4m_2, op4m_3, op4m_4, op4m_5, op4m_6, op4m_7]
      else:
        self.ops+= [lambda x: x.flip((ix, ix+1)), lambda x: x.flip((ix+1,)), lambda x: x.flip((ix,)), \
                  lambda x: x.swapaxes(ix,ix+1).flip((ix+1,)), lambda x: x.swapaxes(ix,ix+1).flip((ix,)), lambda x: x.swapaxes(ix,ix+1).flip((ix,ix+1)), \
                  lambda x: x.swapaxes(ix,ix+1)]
    else:
      raise ValueError("Unknown 2d point group "+self.name)

  def init3d(self):
    self.ops = [lambda x: x]
    ix = 3 if self.channel_first else 2
    if self.name in ['', '1', None, 'C1']:
      self.name = '1'
    elif self.name == 'Oh':
# to generate the operators:
# channel last:
# import numpy as np; import itertools; ord=[1,-1]; print(",\n".join(["lambda x: np.transpose(x,"+str((0,1)+ perm + (5,)) + ")[:,:," + (",".join(["::"+str(m)   for m in mirr]))+"]" for perm in list(itertools.permutations(range(2,5)))  for mirr in itertools.product(ord, ord, ord)]))
# channel first:
# import numpy as np; import itertools; ord=[1,-1]; print(",\n".join(["lambda x: np.transpose(x,"+str((0,1,2)+ perm) + ")[:,:,:," + (",".join(["::"+str(m)   for m in mirr]))+"]" for perm in list(itertools.permutations(range(3,6)))  for mirr in itertools.product(ord, ord, ord)]))
##### Updated to work with Pytorch
# import numpy as np; import itertools; ord=[0,1]; print(",\n".join(["lambda x: x.permute("+str(tuple(range(ix))+ perm + tuple(range(ix+3,6))) + f").flip({(np.nonzero(mirr)[0]+ix).tolist()})" for perm in list(itertools.permutations(range(ix,ix+3)))  for mirr in itertools.product(ord, ord, ord)]).replace(".flip([])",""))
      if self.channel_first:
        self.ops+= [
# lambda x: x.permute((0, 1, 2, 3, 4, 5)),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([5]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([4]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3, 4, 5]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([5]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([4]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([3]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 3, 5, 4)).flip([3, 4, 5]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([5]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3, 4, 5]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([5]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([4]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([3]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 4, 5, 3)).flip([3, 4, 5]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([5]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([4]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([3]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 5, 3, 4)).flip([3, 4, 5]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([5]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([4]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([4, 5]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([3]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([3, 5]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 5, 4, 3)).flip([3, 4, 5])
        ]
      else:
        self.ops+= [
# lambda x: x.permute((0, 1, 2, 3, 4, 5)),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([4]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([2]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 2, 3, 4, 5)).flip([2, 3, 4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([2]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 2, 4, 3, 5)).flip([2, 3, 4]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([4]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([3]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([2]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 3, 2, 4, 5)).flip([2, 3, 4]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([4]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([3]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([2]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 3, 4, 2, 5)).flip([2, 3, 4]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([4]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([3]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([2]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 4, 2, 3, 5)).flip([2, 3, 4]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([4]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([3]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([3, 4]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([2]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([2, 4]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([2, 3]),
lambda x: x.permute((0, 1, 4, 3, 2, 5)).flip([2, 3, 4])
        ]
# # lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 4, 5))[:,:,:,::-1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 3, 5, 4))[:,:,:,::-1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 3, 5))[:,:,:,::-1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 4, 5, 3))[:,:,:,::-1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 3, 4))[:,:,:,::-1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::1,::-1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::-1,::1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::-1,::1,::-1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::-1,::-1,::1],
# lambda x: np.transpose(x,(0, 1, 2, 5, 4, 3))[:,:,:,::-1,::-1,::-1]]
    else:
      raise ValueError("Unknown 3d point group "+self.name)


  @property
  def nops(self):
    """Returns the total number of ops """
    return len(self.ops)

  def random_op(self):
    return np.random.choice(self.ops)

  def random_nontrivial_op(self):
    return np.random.choice(self.ops[1:])

  def __call__(self, x):
    if self.nops > 1:
      return self.random_op()(x)
    else:
      return x

