import subprocess
import os, glob
import tempfile
import numpy as np
from contextlib import contextmanager

def co(instr, split=False):
    out=subprocess.Popen(instr, stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()[0]
    return out.split('\n') if split else out


@contextmanager
def temp_txt_file(data):
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wt')
    temp.write(data)
    temp.close()
    try:
        yield temp.name
    finally:
        os.unlink(temp.name)


def save_traj(fname, traj_arrs, symbols=None, CoM=False):
    if not isinstance(symbols, (list, tuple)):
        symbols = [symbols] * len(traj_arrs)
    if False:# isinstance(traj_arrs[0][0], (int, float)):
        import torch
        traj_arrs = torch.split(torch.tensor(traj_arrs), [len(s) for s in symbols])
    if CoM:
        traj_arrs = [f - torch.mean(f, 0, keepdim=True) for f in traj_arrs]
    from NPS_common.periodic_table import symbol_from_Z
    try:
        symbols = [symbol_from_Z(s) for s in symbols]
    except:
        pass
    if fname.endswith('.xyz'):
        from ase import Atoms
        from ase.io.xyz import write_xyz
        with open(fname, 'w') as fh:
            write_xyz(fh, [Atoms(symbol, positions=arr) for arr, symbol in zip(traj_arrs, symbols)])


from collections import namedtuple
MDFrame = namedtuple('MDFrame', ('positions', 'velocities', 'box', 'node_type', 'time'))

def read_traj(f):
    if f.endswith('.pdb'):
        import MDAnalysis
        return MDAnalysis.coordinates.PDB.PDBReader(f).trajectory
    elif f.endswith(('.pdb', '.xyz', '.extxyz', '.cif')):
        import ase.io
        return ase.io.read(f, index=':')
    elif ('.lammpstraj' in f) or ('.lammpstrj' in f):
        from ase.io.lammpsrun import read_lammps_dump
        return read_lammps_dump(f, index=slice(0,None))
    elif f.endswith('.xtc'):
        path = os.path.dirname(f)
        top = sorted(glob.glob(f'{path}/atom_type.txt'))
        try:
            node_type = list(filter(bool, np.genfromtxt(top[0], dtype='str')))
        except:
            print(f"*********** WARNING: ***********\nCannot find atom_type.txt next to {f}")
            node_type = None
        from MDAnalysis.lib.formats.libmdaxdr import XTCFile
        with XTCFile(f) as f_traj:
            # note: convert nm to Angstrom
            traj = [MDFrame(frame.x*10, None, frame.box, node_type if node_type is not None else np.zeros(len(frame.x), dtype=int), frame.time) for frame in f_traj]
        if len(traj) >= 2:
            if traj[0].time == traj[1].time: # modified xtc format with pos, velocity
                print(f'Found velocity in XTC {f}')
                traj = [MDFrame(traj[i].positions, traj[i+1].positions, traj[i].box, traj[i].node_type, traj[i].time) for i in range(0, len(traj)//2*2, 2)]
        return traj
    else:
        raise ValueError(f'Unknown trajectory type in {f}')


def read_topol(f):
    from itertools import permutations

    lines = open(f,'r').read().splitlines()#open(f, 'r').readlines()
    lines = list(filter(lambda x: not x.startswith(';'), map(lambda x: x.strip(), lines)))
    # print(f'debug l 3', lines[:13])
    keys = ('atom', 'bond', 'pair', 'angle', 'dihedral')
    order = (1, 2, 2, 3, 4)
    istart = [lines.index(f'[ {tag}s ]') for tag in keys]
    iend = (np.array(istart)[1:]).tolist()
    # print(iend, lines.index('', istart[-1]+1))
    iend.append(lines.index('', istart[-1]))
    g = {k: np.array(list(map(lambda x: list(map(int, x.split()[:order[i]])), filter(bool, lines[istart[i]+1:iend[i]]))))-1 for i, k in enumerate(keys)}
    nnode = len(g['atom'])
    g['charge'] = np.array(list(map(lambda x: list(map(float, x.split()[6:7])), filter(bool, lines[istart[0]+1:iend[0]]))))
    bond = np.concatenate((g['bond'], g['bond'][:,::-1]))
    g['bond_index'] = bond.T
    g['bond_type'] = np.ones_like(g['bond_index'][0])
    # print(bond.shape, g['bond_index'].shape, g['bond_type'].shape )
    # bond_idx = np.arange(nnode**2).reshape(-1,nnode)
    bond_idx = np.zeros([nnode]*2, dtype=int)
    bond_idx[tuple(bond.T)] = np.arange(len(bond))
    # angle = [[bond_idx[jk[0], i], bond_idx[jk[1], i]] for i in range(nnode) for jk in permutations(bond[:,1][bond[:,0]==i], 2)]
    angle_index = np.array([[bond_idx[i,j], bond_idx[k,j]] for i,j,k in g['angle']])
    angle_index = np.concatenate((angle_index, angle_index[:,::-1]))
    g['angle_idx'] = angle_index
    dihedral_index = np.array([[bond_idx[i,j], bond_idx[l,k]] for i,j,k,l in g['dihedral']])
    dihedral_index = np.concatenate((dihedral_index, dihedral_index[:,::-1]))
    g['dihedral_idx'] = dihedral_index
    # print(istart, iend, [lines[i:j] for i, j in zip(istart, iend)])
    # print(g, "g['angle_idx']", g['angle_idx'].shape, g['angle_idx'])
    return g



def load_nequip_dataset(fname, pbc=False):
    import ase
    ds = np.load(fname)
    if not pbc:
        return [ase.Atoms(#ds['name'], 
        positions=pos, symbols=None, numbers=ds['z'], pbc=False, cell=[0,0,0]) for pos in ds['R']]
