__author__ = 'Fei Zhou'

import os
import numpy as np
import torch
from torch_geometric.data import Data
# import pickle
from NPS_common.io_utils import read_traj
from NPS_common.utils import unique_list_str
from NPS_common.graph_utils import atoms2pygdata
import ase.io
from io import StringIO


def register_args(parser):
    parser.add_argument('--data_suffix', type=str, default='csv', help='data file type: csv')
    parser.add_argument('--data_format', type=str, default='cif', help='data format: cif')
    parser.add_argument('--extra_keys', type=str, default='', help='comma separated list, e.g. "v", "f", "angle_idx", "dihedral_idx"')
    parser.add_argument('--graph_processing_cutoff', type=float, default=-1, help='If positive, process data as PeriodicRadiusGraph')

def post_process_args(args):
    args.extra_keys = list(filter(bool, args.extra_keys.split(',')))

class graph_single():
    def __init__(self, args, datf):
        self.args = args
        self.dim = args.dim
        self.statistics = {}
        flat = self.load_data(datf)
        if args.data_slice:
            print('data slicing with', args.data_slice)
            flat = eval(f'flat[{args.data_slice}]')
            print(f"Keeping {len(flat)} points")
        self.dataset_postprocess(flat)

    def dataset_preprocess(self, args, data, **kwx):
        return data

    def dataset_postprocess(self, data, to_torch=True, to_id=True, to_graph=True, **kwx):
        flat = atoms2pygdata(data)
        if self.args.graph_processing_cutoff > 0:
            from graphite.transforms.radius_graph_ase import RadiusGraph_ase
            # from graphite.transforms import PeriodicRadiusGraph
            transform = RadiusGraph_ase(cutoff=self.args.graph_processing_cutoff)
            flat = [transform(d) for d in flat]
        self.flat = flat
        self.statistics = {}
        return
        keys = ['pos'] + self.args.extra_keys
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
        import glob
        suffix = self.args.data_suffix
        pts_all = []
        # extra_keys = self.args.extra_keys
        for file in sorted(glob.glob(f+f'*.{suffix}')):
            # from pymatgen.core.structure import Structure
            if file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file)
                # points = [Structure.from_str(x, fmt='cif') for x in df['cif']]
                points = [ase.io.read(StringIO(x), format='cif') for x in df['cif']]
            else:
                points = read_traj(file)
            print(f'  reading {file} got', len(points))
            shift_center_of_mass = True
            if shift_center_of_mass and (not self.args.periodic):
                for a in points:
                    a.positions -= np.mean(a.positions, axis=0, keepdims=True)
            pts_all.extend(points)
        print(f'Loaded {len(pts_all)} from {f}')
        return pts_all

    def __getitem__(self, i):
        # j = self.start_pos[i]
        return self.flat[i]

    def __len__(self):
        return len(self.flat)

if __name__ == '__main__':
    from collections import namedtuple
    data_args = {"data_suffix":"cif", 'dim':3}
    data_args = namedtuple('Data_args_init', data_args.keys())(*data_args.values())
    ds = graph_single(data_args, "/g/g90/zhou6/lassen-space/cdvae/data/carbon_24/train.csv")

