import torch
from torch import nn
from functools import partial
from graphite.nn.basis import bessel
# from NPS.model.common import vector_pbc

class InitialEmbedding(nn.Module):
    def __init__(self, num_species, pbc=False, cutoff=4.0, precompute_bond_vec_correction=False):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
        self.pbc = pbc
        self.precompute_bond_vec_correction = precompute_bond_vec_correction
    
    def forward(self, data):
        # Embed node
        h_node_x = self.embed_node_x(data.x)
        h_node_z = self.embed_node_z(data.x)

        # Must set to make an entry point for pos
        data.edge_attr = data.positions[data.edge_index[1]] - data.positions[data.edge_index[0]]
        if self.pbc:
            if self.precompute_bond_vec_correction:
                data.edge_attr = data.edge_attr + data.edge_vec_correction
            else:
                data.edge_attr -= torch.bmm(torch.round(torch.bmm(data.edge_attr.detach()[:,None], data.inv_lattice[data.edge_index[0]])), data.lattice[data.edge_index[0]])[:,0]
#           data.edge_attr = vector_pbc(data.edge_attr, data.lattice[data.edge_index[0]], data.inv_lattice[data.edge_index[0]])
        # Embed edge
        h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        data.h_node_x = h_node_x
        data.h_node_z = h_node_z
        data.h_edge   = h_edge
        return data
