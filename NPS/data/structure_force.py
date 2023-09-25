__author__ = 'Fei Zhou'

import os
import numpy as np
import torch
from torch_geometric.data import Data
# import pickle
# from NPS_common.utils import unique_list_str
# from io import StringIO
from GNN.MeshGraphNets.common import get_neighbor_list
from NPS.data.graph_single import graph_single
import ase.io

def register_args(parser):
    parser.add_argument('--data_suffix', type=str, default='extxyz', help='data file type')
    # parser.add_argument('--data_cutoff', type=float, default=-1., help='pre-compute edges')
    parser.add_argument('--data_Eref', type=str, default="", help='Example "8:-10.0,1:-3.1"')
    parser.add_argument('--data_atomtypes', type=str, default="", help='Example "O,H"')

def post_process_args(args):
    pass

def read_array_flattened(shape_str, arr_str):
    shape = list(map(int, filter(None, shape_str.split())))
    return np.fromstring(arr_str, dtype=float, sep='\t').reshape(shape)

class structure_force(graph_single):
    # def __init__(self, args, datf):
    #     self.args = args
    #     # self.dim = args.dim
    #     self.statistics = {}
    #     self.onsiteOnly = self.args.onsiteOnly
    #     print("hubbard1band_chempot loading")
    #     flat = self.load_data(datf)
    #     print("hubbard1band_chempot loaded")
    #     self.dataset_postprocess(flat)

    def load_data(self, f):
        import glob
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.args.data_atomtypes.split(','))
        suffix = self.args.data_suffix
        pts_all = []
        if self.args.data_Eref:
            Eref_dict = eval("{"+self.args.data_Eref+"}")
            Eref = lambda zs: np.sum([Eref_dict[z] for z in zs])
        else:
            Eref = lambda zs: 0
        for fn in sorted(glob.glob(f+('/' if os.path.isdir(f) else '')+f'*{suffix}')):
            structures = ase.io.read(fn, index=":")
            nStr = len(structures)
            print(f'  reading {fn} got {nStr} configurations')
            for i, s in enumerate(structures):
                d = {
                     "positions": torch.from_numpy(s.positions).float(),
                     "x": torch.from_numpy(le.transform(s.get_chemical_symbols())),
                     "cell": torch.from_numpy(s.cell[:]).float(),
                     "pbc": torch.from_numpy(s.pbc),
                     "forces": torch.from_numpy(s.get_forces()).float(),
                     "energy": torch.tensor([[s.get_potential_energy()-Eref(s.get_atomic_numbers())]], dtype=torch.float32),
                     "num_nodes": s.positions.shape[0],
                    #  "node_y": torch.zeros((properties[0].shape[0], 7)),
                    #  "node_features": torch.zeros((properties[0].shape[0], 5)),
                     }
                if self.args.edge_cutoff > 0:
                    idx, edges = get_neighbor_list(d["positions"], s.cell[:], d["pbc"], self.args.edge_cutoff)
                    d["edge_index"] = idx.long()
                    d["edge_vec_correction"] = (edges-(s.positions[idx[1]]-s.positions[idx[0]])).float()
                pts_all.append(Data.from_dict(d))
        print(f'Loaded {len(pts_all)} from {f}')
        self.flat = pts_all
        return

    def dataset_postprocess(self, data, to_torch=True, to_graph=True, **kwx):
        pass



if __name__ == '__main__':
    from collections import namedtuple
    data_args = {"data_suffix":".extxyz", 'edge_cutoff':2., "data_atomtypes":"Hf,O",
                 "data_Eref":"8:-9.92,72:-9.92", "dim":3}
    data_args = namedtuple('Data_args_init', data_args.keys())(*data_args.values())
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, default="/g/g90/zhou6/amsdnn/data/a-HfO2/valid", help="data dir")
    options = parser.parse_args()
    ds = structure_force(data_args, options.dir)
    print(ds[99]); print(ds[::99], ds[99].energy)
#     from torch_scatter import scatter
#     out_dat = torch.cat([
#         torch.cat([
#             torch.cat([scatter(pt.edge_features, pt.edge_index[0], dim=0, reduce=r)
#       for r in ("sum", "mean", "min", "max")] +
#       [1/scatter(1/pt.edge_features, pt.edge_index[0], dim=0, reduce=r)
#       for r in ("sum", "mean")]
#       , dim=-1),
#             pt.node_features[:,0:1],
#             pt.node_features[:,1:2]+pt.node_features[:,4:5],
#             pt.node_features[:,1:2]-pt.node_features[:,4:5],
#             pt.node_y[:,0:1]+pt.node_y[:,3:4],
#             pt.node_y[:,0:1]-pt.node_y[:,3:4],
#             pt.node_y[:,[4,5,6]],
#         ], dim=-1)
#         for pt in ds
#         for _ in range(3)
#     ], dim=0)
#     np.save('tmp.npy', out_dat.numpy().astype(np.float32))
#     #"/g/g90/zhou6/amsdnn/data/hubbard1band/onsite1/COL1-muMax1-UMin0-UMax0-tMin0-tMax1/train/*.txt")

