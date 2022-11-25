import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd import Variable

NNop = {(2,'conv'): nn.Conv2d, (2,'bnorm'): nn.BatchNorm2d, (2, 'AdaptiveAvgPool'): nn.AdaptiveAvgPool2d,
        (3,'conv'): nn.Conv3d, (3,'bnorm'): nn.BatchNorm3d, (3, 'AdaptiveAvgPool'): nn.AdaptiveAvgPool3d,
        (1,'conv'): nn.Conv1d, (1,'bnorm'): nn.BatchNorm1d, (1, 'AdaptiveAvgPool'): nn.AdaptiveAvgPool1d,
        (1,'convtr'): nn.ConvTranspose1d, (2,'convtr'): nn.ConvTranspose2d, (3,'convtr'): nn.ConvTranspose3d,
        (1,'conv_op'): F.conv1d, (2,'conv_op'): F.conv2d, (3,'conv_op'): F.conv3d
       }

def layer_norm_optional(shape, flag, **kwargs):
    if flag == 0:
        return nn.Identity()
    elif flag == 1:
        return nn.LayerNorm(shape, **kwargs)
    elif flag == -1:
        return nn.LayerNorm(shape, elementwise_affine=False, **kwargs)
    else:
        raise ValueError(f'Unknown layer norm flag {flag}')

# # def MYconv(dim=2, periodic=False):
# #     op = NNop[(dim,'conv')]
# #     pad = 'circular' if periodic else 'zeros'
# #     return lambda *args, **kwargs: op(*args, padding_mode=pad, **kwargs)

# def default_conv(in_channels, out_channels, kernel_size, bias=True, dim=2, periodic=False, stride=1, kernel=None):
#     if kernel is not None:
#         assert (bias==False) and (stride==1)
#         return lambda x: NNop[(dim,'conv_op')](
#         F.pad(x, 1, mode='circular'), kernel)
#     return NNop[(dim,'conv')](
#         in_channels, out_channels, kernel_size,
#         padding_mode='circular' if periodic else 'zeros',
#         padding=(kernel_size//2), bias=bias, stride=stride)

class Rec_Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x).clamp(min=0)

my_activations={'relu':nn.ReLU(), 'prelu':nn.PReLU(), 'lrelu':nn.LeakyReLU(), 'rec_tanh':Rec_Tanh(),
#    'mish':nn.Mish(),
    'gelu': nn.GELU(),
    'swish':nn.SiLU(), 'silu':nn.SiLU(),
    'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh()}
try:
    my_activations['mish'] = nn.Mish()
except:
    class MyMish(nn.Module):
        def forward(self, inputs):
            return inputs * torch.tanh(F.softplus(inputs))
    my_activations['mish'] = MyMish()



def shift_to_const(x, x_target, flags):
    x_mean = torch.mean(x, tuple(range(2,x.ndim)), True)
    return x + flags*(-x_mean+x_target)


class GradientNorm(nn.Module):
    def forward(self, x):
        return torch.norm(Gradient(x), dim=1)


import functools
@functools.lru_cache(10)
def laplacian_stencil(dim=2):
    lapl_stencil = np.zeros((3,)*dim, dtype=np.float32)
    lapl_stencil[(1,)*dim] = -2*dim
    for i in np.vstack([1-np.eye(dim,dtype=int), 1+np.eye(dim,dtype=int)]):
        lapl_stencil[tuple(i)]=1
    return lapl_stencil[None,None,...]


def gradient_stencil(dim=2, offset=1):
    stencil = np.zeros((dim,1)+(3,)*dim, dtype=np.float32)
    for ii, i in enumerate(np.eye(dim,dtype=int)):
        if offset==0:
            stencil[(ii,0)+tuple(1-i)]=-0.5
            stencil[(ii,0)+tuple(1+i)]=0.5
        elif offset==1:
            stencil[(ii,0)+tuple(1-i*0)]=-1
            stencil[(ii,0)+tuple(1+i)]=1
        elif offset==-1:
            stencil[(ii,0)+tuple(1-i*0)]=1
            stencil[(ii,0)+tuple(1-i)]=-1
    return stencil


def average_perdir_stencil(dim=2, offset=1):
    return np.abs(gradient_stencil(dim, offset)) * (1 if offset==0 else 0.5)


def divergence_stencil(dim=2, offset=1):
    stencil = np.zeros((1,dim)+(3,)*dim, dtype=np.float32)
    for ii, i in enumerate(np.eye(dim,dtype=int)):
        if offset==0:
            stencil[(0,ii)+tuple(1-i)]=-0.5
            stencil[(0,ii)+tuple(1+i)]=0.5
        elif offset==-1:
            stencil[(0,ii)+tuple(1-i*0)]=-1
            stencil[(0,ii)+tuple(1+i)]=1
        elif offset==1:
            stencil[(0,ii)+tuple(1-i*0)]=1
            stencil[(0,ii)+tuple(1-i)]=-1
    return stencil

class ConvFixedKer(nn.Module):
    def __init__(self, kernel, dim=2, periodic=True, padding=True, **kwargs):
        super(ConvFixedKer, self).__init__()
        self.conv = {1:F.conv1d,2:F.conv2d,3:F.conv3d}[dim]
        self.dim = dim
        self.register_buffer('ker', torch.tensor(kernel, requires_grad=False)) #,persistent=False  #in_channel, out_channel,, device='cuda')
        self.padding = (kernel.shape[-1]//2,kernel.shape[-1]//2)*self.dim
        self.mode = 'circular' if periodic else 'zeros'
        self.to_pad = padding and np.any(self.padding)
        self.conv_kwargs = kwargs

    def forward(self, x):
        if self.to_pad:
            return self.conv(F.pad(x, self.padding, mode=self.mode), self.ker, **self.conv_kwargs)
        else:
            return self.conv(x, self.ker, **self.conv_kwargs)

def Laplacian_Conv(dim=2, periodic=True, padding=True):
    assert periodic, ValueError('ERROR Laplacian using convolution is implemented only for periodic condition')
    return ConvFixedKer(laplacian_stencil(dim), dim, periodic, padding)

def Gradient_Conv(dim=2, periodic=True, padding=True, offset=1):
    assert periodic, ValueError('ERROR Gradient using convolution is implemented only for periodic condition')
    return ConvFixedKer(gradient_stencil(dim, offset=offset), dim, periodic, padding)

def Divergence_Conv(dim=2, periodic=True, padding=True, offset=1):
    assert periodic, ValueError('ERROR Divergence using convolution is implemented only for periodic condition')
    return ConvFixedKer(divergence_stencil(dim, offset=offset), dim, periodic, padding)

def Average_PerDir_Conv(dim=2, periodic=True, padding=True, offset=1):
    """
    For n-dim, return n channels. For direction i, a(r)+a(r+i), i.e. same as gradient but with '-'=>'+'
    """
    return ConvFixedKer(average_perdir_stencil(dim, offset=offset), dim, periodic, padding)

class Divergence_Gradient_symm(nn.Module):
    def __init__(self, dim=2, periodic=True, padding=True, offset=1):
        super().__init__()
        assert offset != 0
        self.gradient = Gradient_Conv(dim=dim, periodic=periodic, padding=padding, offset=offset)
        self.divergence = Divergence_Conv(dim=dim, periodic=periodic, padding=padding, offset=offset)
        self.average_dir = Average_PerDir_Conv(dim=dim, periodic=periodic, padding=padding, offset=offset)

    def forward(self, M, mu):
        return self.divergence(self.average_dir(M) * self.gradient(mu))
#        return ((roll(a,-1,0)+a)/2)*(roll(b,-1,0)-b) - ((a+roll(a,1,0))/2)*(b-roll(b,1,0)) + ((roll(a,-1,1)+a)/2)*(roll(b,-1,1)-b) - ((a+roll(a,1,1))/2)*(b-roll(b,1,1))


# class Laplacian_Conv(nn.Module):
#     def __init__(self, dim=2, periodic=True):
#         super(Laplacian_Conv, self).__init__()
#         assert periodic, ValueError('ERROR Laplacian using convolution is implemented only for periodic condition')
#         self.conv = {1:F.conv1d,2:F.conv2d,3:F.conv3d}[dim]
#         self.dim = dim
#         self.register_buffer('ker', torch.tensor(laplacian_stencil(self.dim), requires_grad=False)) #,persistent=False  #in_channel, out_channel,, device='cuda')
#         self.padding = (1,1)*self.dim

#     def forward(self, x):
#         return self.conv(F.pad(x, self.padding, mode='circular'), self.ker)


def laplacian_roll(x, lvl=0, dim=2, type='np'):
    theroll = np.roll if type=='np' else (roll if type=='pt' else tf.roll)
    if dim==1:
        return theroll(x,1,0+lvl)+theroll(x,-1,0+lvl)-2*x
    elif dim==2:
        return (theroll(x,1,0+lvl)+theroll(x,-1,0+lvl)+theroll(x,1,1+lvl)+theroll(x,-1,1+lvl))-4*x
    elif dim==3:
        return (theroll(x,1,0+lvl)+theroll(x,-1,0+lvl)+theroll(x,1,1+lvl)+theroll(x,-1,1+lvl)+theroll(x,1,2+lvl)+theroll(x,-1,2+lvl))-6*x
    elif dim==4:
        return (theroll(x,1,0+lvl)+theroll(x,-1,0+lvl)+theroll(x,1,1+lvl)+theroll(x,-1,1+lvl)+theroll(x,1,2+lvl)+theroll(x,-1,2+lvl)+
          theroll(x,1,3+lvl)+theroll(x,-1,3+lvl))-8*x
    elif dim==5:
        return (theroll(x,1,0+lvl)+theroll(x,-1,0+lvl)+theroll(x,1,1+lvl)+theroll(x,-1,1+lvl)+theroll(x,1,2+lvl)+theroll(x,-1,2+lvl)+
          theroll(x,1,3+lvl)+theroll(x,-1,3+lvl)+theroll(x,1,4+lvl)+theroll(x,-1,4+lvl))-10*x

def CNN(D_in, Hs, D_out, kernel, activation='relu', dim=2, periodic=False, last_bias=True):
    """
    kernel can be an integer, or a list of ints for each layer
    """
    act= my_activations[activation]
    ns = [D_in] + Hs + [D_out]
    ks = [kernel]*(len(Hs)+1) if isinstance(kernel, int) else kernel
    if len(ks)==1: ks*=len(Hs)+1
    m = [x for i in range(len(Hs)+1) for x in [ConvND(ns[i], ns[i+1], ks[i], dim, periodic=periodic, bias=(i<len(Hs) or last_bias)), act]]
    return nn.Sequential(*m[:-1])


class TwoPointDiffusivity(nn.Module):
    def __init__(self, D_in, Hs, D_out, kernel, activation='relu', dim=2, periodic=False, neighbor='NN', offset=1):
        super().__init__()
        self.dim=dim
        self.neighbor=neighbor
        self.offset=offset
        self.net = CNN(D_in*2, Hs, D_out, kernel, activation=activation, dim=dim, periodic=periodic)

    def forward(self, c, one_point_test=True):
        if one_point_test:
            return self.net(torch.cat([c, c],dim=1))*2
        else:
            # symmetric two point function for diffusivity
            return torch.cat([self.net(torch.cat([c, torch.roll(c,-self.offset,d+2)],dim=1))+
                              self.net(torch.cat([torch.roll(c,-self.offset,d+2), c],dim=1)) for d in range(self.dim)], dim=1)

def MLP_(D_in, Hs, D_out, activation='sigmoid', conv=True, dim=2, dropout=-0.5,
  dropout_first=False,
  activate_final=False, layer_norm=False):
    """ dim: dimension of image if conv """
    act= my_activations[activation] if isinstance(activation, str) else activation
    ns = [D_in] + Hs + [D_out]
    m = [x for i in range(len(Hs)+1) for x in ([nn.Dropout(dropout)] if (dropout>0) and (i>0 or dropout_first) else [])+ [NNop[(dim,'conv')](ns[i],ns[i+1],1,bias=(i<len(Hs))) if conv else nn.Linear(ns[i],ns[i+1], bias=(i<len(Hs))), act]]
    if not activate_final:
        m = m[:-1]
    if layer_norm:
        m.append(nn.LayerNorm(D_out)) # place holder, TBD
    return nn.Sequential(*m)

class MLP(nn.Module):
    def __init__(self, D_in, H, D_out, activation='sigmoid', conv=True, dim=2, symmetric=False, squared=False, dropout=-0.5,
      activate_final=False, layer_norm=False):
        super(MLP, self).__init__()
        self.symmetric = symmetric
        self.squared = squared
        self.conv = conv
        self.net = MLP_(D_in, H, D_out, activation, conv, dim, dropout=dropout, activate_final=activate_final, layer_norm=layer_norm)

    def forward(self, x):
        y = self.net(x)
        if self.symmetric:
            y = (y+ self.net(torch.flip(x,[1 if self.conv else -1])))/2
        if self.squared:
            y = y**2
        return y


def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def gradient(x, dim=2, periodic=True, lvl=1):
    if periodic:
        return gradient_roll(x, lvl=lvl, dim=dim, type='pt')
    else:
        return gradient_conv(x, dim, periodic)

def divergence(x, dim=2, periodic=True, lvl=1):
    if periodic:
        return divergence_roll(x, lvl=lvl, dim=dim, type='pt')
    else:
        return divergence_conv(x, dim, periodic)

def gradient_roll(x, lvl=0, dim=2, type='np'):
    theroll = roll_op(type)
    thecat = cat_op(type)
    if dim==1:
        return (theroll(x,-1,0+lvl)-theroll(x,1,0+lvl))/2
    elif dim==2:
        return thecat([theroll(x,-1,0+lvl)-theroll(x,1,0+lvl),theroll(x,-1,1+lvl)-theroll(x,1,1+lvl)], -1)/2
    elif dim==3:
        return thecat([theroll(x,-1,0+lvl)-theroll(x,1,0+lvl),theroll(x,-1,1+lvl)-theroll(x,1,1+lvl),theroll(x,-1,2+lvl)-theroll(x,1,2+lvl)], -1)/2
    elif dim==4:
        return thecat([theroll(x,-1,0+lvl)-theroll(x,1,0+lvl),theroll(x,-1,1+lvl)-theroll(x,1,1+lvl),theroll(x,-1,2+lvl)-theroll(x,1,2+lvl),
          theroll(x,-1,3+lvl)-theroll(x,1,3+lvl)], -1)/2
    elif dim==5:
        return thecat([theroll(x,-1,0+lvl)-theroll(x,1,0+lvl),theroll(x,-1,1+lvl)-theroll(x,1,1+lvl),theroll(x,-1,2+lvl)-theroll(x,1,2+lvl),
          theroll(x,-1,3+lvl)-theroll(x,1,3+lvl),theroll(x,-1,4+lvl)-theroll(x,1,4+lvl)], -1)/2


def divergence_roll(x, lvl=0, dim=2, type='np'):
    theroll = roll_op(type)
    if dim==1:
        return (theroll(x,-1,0+lvl)-theroll(x,1,0+lvl))/2
    elif dim==2:
        return (theroll(x[...,0:1],-1,0+lvl)-theroll(x[...,0:1],1,0+lvl)+theroll(x[...,1:2],-1,1+lvl)-theroll(x[...,1:2],1,1+lvl))/2
    elif dim==3:
        return (theroll(x[...,0:1],-1,0+lvl)-theroll(x[...,0:1],1,0+lvl)+theroll(x[...,1:2],-1,1+lvl)-theroll(x[...,1:2],1,1+lvl)+
          theroll(x[...,2:3],-1,2+lvl)-theroll(x[...,2:3],1,2+lvl))/2
    elif dim==4:
        return (theroll(x[...,0:1],-1,0+lvl)-theroll(x[...,0:1],1,0+lvl)+theroll(x[...,1:2],-1,1+lvl)-theroll(x[...,1:2],1,1+lvl)+
          theroll(x[...,2:3],-1,2+lvl)-theroll(x[...,2:3],1,2+lvl)+theroll(x[...,3:4],-1,3+lvl)-theroll(x[...,3:4],1,3+lvl))/2
    elif dim==5:
        return (theroll(x[...,0:1],-1,0+lvl)-theroll(x[...,0:1],1,0+lvl)+theroll(x[...,1:2],-1,1+lvl)-theroll(x[...,1:2],1,1+lvl)+
          theroll(x[...,2:3],-1,2+lvl)-theroll(x[...,2:3],1,2+lvl)+theroll(x[...,3:4],-1,3+lvl)-theroll(x[...,3:4],1,3+lvl)+
          theroll(x[...,4:5],-1,4+lvl)-theroll(x[...,4:5],1,4+lvl))/2

def roll_op(type):
    return np.roll if type=='np' else (roll if type=='pt' else tf.roll)

def cat_op(type):
    return np.concatenate if type=='np' else (torch.cat if type=='pt' else tf.concat)


def ConvND(in_channels, out_channels, kernel_size, dim=2, periodic=False, space2d=False, **kwargs):
    assert kernel_size%2 == 1
    assert not (space2d and (dim != 3)), "ERROR when periodic condition for 2 space dimension specified, dim should be 3"
    npad = kernel_size//2 #int(np.ceil((kernel_size-1)/2))
    if space2d:
        # return nn.Sequential(*m)
        raise ValueError('space2d periodic conv not implemented yet')
    else:
        return NNop[(dim,'conv')](in_channels, out_channels, kernel_size,
            padding_mode='circular' if periodic else 'zeros', padding=npad, **kwargs)


class ConvTransposeCircularPad(nn.Module):
    def __init__(self, dim=2, kernel_size=1, stride=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.pad = (kernel_size//2,)*(2*dim)
        self.p = stride*(kernel_size//2)
        self.conv = NNop[(dim,'convtr')](**kwargs, kernel_size=kernel_size,
          stride=stride, padding=kernel_size//2, output_padding=stride-1)

    def forward(self, x):
        if self.p > 0:
            x = F.pad(x, self.pad, mode='circular')
        out = self.conv(x)
        # print(f'debug ker {self.kernel_size} dim {self.dim}')
        if self.p > 0:
            if self.dim == 2:
                out = out[...,self.p:-self.p,self.p:-self.p]
            elif self.dim == 3:
                out = out[...,self.p:-self.p,self.p:-self.p,self.p:-self.p]
            elif self.dim == 1:
                out = out[...,self.p:-self.p]
        return out


def ConvTransposeND(in_channels, out_channels, kernel_size, stride=1, dim=2, periodic=False, **kwargs):
    assert kernel_size%2 == 1
    if periodic:
        npad = kernel_size-1
        # manually pad with periodic images
        return ConvTransposeCircularPad(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, dim=dim, **kwargs)
    else:
        return NNop[(dim,'convtr')](in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size//2, output_padding=stride-1, **kwargs)

def vector_pbc(vec, lattice, inv_lattice=None):
    """ Convert an input vector to shortest image within periodic boundary condition """
    if inv_lattice is None:
        return vec - torch.round(vec / lattice) * lattice
    else:
        return vec - torch.round(vec @ inv_lattice) @ lattice


def diff_filter(dim=3, ker=3, nc=1):
    """Input of shape [nc, n1...ndim], find difference tween neighbor and site """
    npad = (ker-1)//2
    ker_wt = np.zeros((ker**dim,) + (ker,)*dim, dtype=np.float32)
    grid = np.mgrid[(slice(0,ker),)*dim].reshape(dim, -1)#.transpose(list(range(1,1+dim))+[0,])
    grid = np.concatenate((np.arange(grid.shape[1])[None,:], grid), 0)
    ker_wt[tuple([x for x in grid])] = 1
    ker_wt[(slice(0, None),) + (npad,)*dim] -= 1
    return np.tile(ker_wt[:,None], (nc, 1)+ (1,)*dim)


def diff_conv(dim, ker, nc, periodic=False):
    ker_wt = diff_filter(dim=dim, ker=ker, nc=nc)
    return ConvFixedKer(ker_wt, dim=dim, periodic=periodic, groups=nc)


