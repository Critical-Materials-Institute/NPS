__author__ = 'Fei Zhou'

import os
import numpy as np
import torch
from torch_geometric.data import Data
# import pickle
from NPS.data.longclip import longclip
from NPS_common.io_utils import co, read_traj, read_topol
from NPS_common.utils import unique_list_str


def register_args(parser):
    parser.add_argument('--data_suffix', type=str, default='pdb', help='data file type: pdb|lammpstraj|xtc')
    parser.add_argument('--extra_keys', type=str, default='', help='comma separated list, e.g. "v", "f", "angle_idx", "dihedral_idx"')

def post_process_args(args):
    args.extra_keys = list(filter(bool, args.extra_keys.split(',')))

class graph_clip(longclip):
    def dataset_preprocess(self, args, data, **kwx):
        return data

    def dataset_postprocess(self, args, data, to_torch=True, to_id=True, to_graph=True, **kwx):
        self.flat = [y for x in data for y in x]
        self.statistics = {}
        keys = ['pos'] + args.extra_keys
        if to_id:
            if self.args.data_suffix == 'pdb':
                # types = [list(map(lambda s: "-".join(s), x['type'])) for x in self.flat]
                import re
                regex = re.compile(r"[1-9]$")
                types = [[regex.sub('n', y[0]) for y in x['type']] for x in self.flat]
            else:
                types = [x['type'] for x in self.flat]
            # type_unique = np.unique(types)
            type_unique = unique_list_str(types)
            type_dict = dict(zip(type_unique, list(range(len(type_unique)))))
            for i, x in enumerate(self.flat):
                x['type'] = np.array([type_dict[t] for t in types[i]])
                x['symbol'] = types
            # print(self.flat[333])
            self.statistics['type_dict'] = type_dict
            print(f"  Done. Found {len(type_unique)} node types")
        if to_torch:
            for i, x in enumerate(self.flat):
                x['type'] = torch.from_numpy(x['type'])
                for k in keys:
                    x[k] = torch.from_numpy(x[k])
                x['fixedbond']  = torch.from_numpy(x['fixedbond']).long()
                x['fixedbond_type']  = torch.from_numpy(x['fixedbond_type']).int()
        if to_graph:
            for i, x in enumerate(self.flat):
                self.flat[i] = Data(
                node_type=x['type'],
                # edge_index=edge_index.T,
                # edge_features=edge_features,
                fixedbond_index=x['fixedbond'], fixedbond_type=x['fixedbond_type'],
                **({k:x[k] for k in keys}))


    def load_data(self, f):
        traj = []
        import glob
        suffix = self.args.data_suffix
        extra_keys = self.args.extra_keys
        for file in sorted(glob.glob(f+f'/*.{suffix}')):
            f_traj = read_traj(file)
            fixedbond_file = os.path.dirname(file)+'/fixedbond.txt'
            if not os.path.exists(fixedbond_file): fixedbond_file = file
            fixedbond = co(f"head -n3 {fixedbond_file}|grep 'REMARK    FIXEDBOND' |sed 's/REMARK    FIXEDBOND//'")
            if fixedbond:
                fixedbond = np.array(list(map(int, fixedbond.strip().split()))).reshape(-1, 3)
                ## both ij and ji
                fixedbond = np.concatenate((fixedbond, np.concatenate((fixedbond[:,:1], fixedbond[:,2:0:-1]), 1)), 0)
                fixedbond, fixedbond_type = fixedbond[:,1:3].T, fixedbond[:,0]
            else:
                fixedbond = np.array([])
                fixedbond_type = np.array([])
            print(f'  reading {file} got', len(f_traj))
            if suffix == 'pdb':
                typ_list = co(f"grep ENDMDL -m1 -B9999999 {file}|grep ATOM |awk "+"""'{print $3,$4,$11}'""")
                typ_list = np.array(typ_list.strip().split()).reshape(-1, 3)
                typ_func = lambda _: typ_list
            elif suffix == 'lammpstraj':
                typ_func = lambda x: x.get_atomic_numbers()
            elif suffix == 'xtc':
                typ_func = lambda x: x.node_type
            extra_dict = {}
            possible_keys = ('bond_index', 'bond_type', 'pair_index', 'angle_idx', 'dihedral_idx')
            if extra_keys:
                if any([s in extra_keys for s in possible_keys]):
                    topol = read_topol(f'{file}_topol.top')
                for k in extra_keys:
                    if k == 'f':
                        forces = np.loadtxt(f'{file}_force.txt').reshape(-1, *f_traj[0].positions.shape).astype(np.float32)
                        extra_dict[k] = forces
                    elif k in possible_keys:
                        extra_dict[k] = [topol[k]]*len(f_traj)
                    else:
                        raise NotImplementedError(f'Reading {k} is NOT implemented yet')
            traj.append([{'pos':np.array(x.positions, dtype=np.float32), 'type':typ_func(x), 'fixedbond':fixedbond, 'fixedbond_type':fixedbond_type,
                **({k:extra_dict[k][i] for k in extra_dict})} for i,x in enumerate(f_traj)])
        return traj

    def __getitem__(self, i):
        j = self.start_pos[i]
        if self.args.n_out == 0: #3 non-sequence
            return self.flat[j]
        return self.flat[j:j+self.tot_len_frame:self.nskip]




if __name__ == '__main__':
    from collections import namedtuple
    data_args = {"data_suffix":"pdb"}
    data_args = namedtuple('Data_args_init', data_args.keys())(*data_args.values())
    ds = graph_clip(data_args, "/g/g90/zhou6/amsdnn/data/Chignolin_traj/Chignolin/xtc", 2)

