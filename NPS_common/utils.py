import os
import numpy as np
import functools
import re


def grid_points(Ns, corner=True):
    # return np.stack(np.meshgrid(*[np.linspace(0, 1, N+1) for N in Ns], indexing='ij'),-1).reshape((-1, len(Ns)))
    pts = np.stack(np.meshgrid(*[np.arange(N+1) for N in Ns], indexing='ij'),-1).reshape((-1, len(Ns)))
    if corner:
        return pts
    else:
        return np.array([x for x in pts if np.any(x%Ns)])


def digits2int(digits, bases):
    """
    digits: e.g. (1, 0)
    bases: e.g. (64, 32)
    returns: 1*32 + 0
    """
    dim = len(bases)
    if dim == 2:
        return digits[:,0]*bases[1] + digits[:, 1]
    elif dim == 3:
        return digits[:,0]*bases[1]*bases[2] + digits[:, 1]*bases[2] + digits[:, 2]


# def grid_points_excl_corner(Ns):
#     return np.array([x for x in grid_points_incl_corner(Ns) if np.any(x%Ns)])

# @functools.lru_cache(maxsize=128, typed=False)
def linear_interp_coeff(Ns):
    dim = len(Ns)
    x0 = np.zeros(dim)
    x1 = np.ones(dim)
    def coeff_(x):
        return functools.reduce(np.outer, np.stack([x1, x],1)).ravel()

    mat = [coeff_(x) for x in grid_points([1,]*dim)]
    # print(np.stack(np.meshgrid(*([0,1]*dim), indexing='ij'),-1).reshape((-1, dim)))
    # print(mat)
    grid = grid_points(Ns)
    grid_pts = [coeff_(x) for x in grid_points(Ns)/Ns]
    return np.dot(grid_pts, np.linalg.inv(mat)).T
    # print(np.meshgrid(*([0,1]*dim), indexing='ij'))


#def a1line(arr): return re.sub('[\[\]]', '', np.array2string(np.array(arr),precision=3,max_line_width=np.inf))
def a1line(arr): return re.sub('[\[\]]', '', np.array2string(np.array(arr),max_line_width=np.inf,\
  formatter={'float_kind':lambda x: "%7.3e" % x}))

def str2list(x, typ=int): return list(map(typ, filter(bool, x.strip().split(','))))


def load_array(fname, todense=True):
    if fname[-4:]=='.bin':
        return np.fromfile(fname, dtype=np.float).astype(np.float32)
    elif fname[-4:]=='.npy':
        return np.load(fname)
    # elif fname[-7:]=='.sp.npz':
    #     import scipy.sparse
    #     dat = scipy.sparse.load_npz(fname)
    #     return dat.toarray() if todense else dat
    elif fname[-7:]=='.sp.npz':
        d = np.load(fname)
        dat = np.zeros(d['shape'], dtype=d['value'].dtype)
        dat[tuple(d['index'])] = d['value']
        return dat
    elif fname[-4:]=='.npz':
        # dense array
        dat = np.load(fname)
        return dat['arr_0']
    elif fname[-4:]=='.txt':
        return np.loadtxt(fname)
    else:
        raise ValueError(f'Unknown format in {fname}')

def save_array(fname, arr):
    if fname[-4:]=='.bin':
        raise 'NOT implemented'
        # return np.fromfile(fname, dtype=np.float).astype(np.float32)
    elif fname[-4:]=='.npy':
        np.save(fname, arr)
    # elif fname[-7:]=='.sp.npz':
    #     import scipy.sparse
    #     scipy.sparse.save_npz(fname, scipy.sparse.csr_array(arr))
    elif fname[-7:]=='.sp.npz':
        nz = np.nonzero(arr)
        np.savez_compressed(fname, shape=arr.shape, index=np.array(nz), value=arr[nz])
    elif fname[-4:]=='.npz':
        # dense array
        np.savez_compressed(fname, arr)
    else:
        raise ValueError(f'Unknown format in {fname}')

def load_array_auto(fn):
    if not os.path.exists(fn):
        if os.path.exists(fn[:-4]+'.sp.npz'):
            fn = fn[:-4]+'.sp.npz'
        elif os.path.exists(fn[:-4]+'.npz'):
            fn = fn[:-4]+'.npz'
        else:
            print(f'WARNING cannot load from {fn}')
    return load_array(fn)


def repr_simple_graph(g):
    def _repr_item(v):
        if hasattr(v, 'shape'):
            return v.shape
        elif isinstance(v, dict):
            return [(k1, v1.shape if hasattr(v1, 'shape') else v1) for k1,v1 in v.items()]
        else:
            return v

    return [(k, _repr_item(v)) for k,v in g.items()]


def str2slice(s, range_only=False):
    s = list(map(lambda x: int(x) if x else None, s.split(':')))
    assert len(s) <= 3
    if len(s) == 1:
        if range_only:
            s = slice(s[0], (s[0]+1 if s[0]!=-1 else None))
        else:
            s = s[0]
    elif len(s) >= 2:
        if s[0] is None: s[0] = 0
        s = slice(*s)#eval(f'slice({options.ichannel})')
    return s


def unique_list_str(list_str):
    import itertools
    return list(sorted(set(itertools.chain.from_iterable(list_str))))

