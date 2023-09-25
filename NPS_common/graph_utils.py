import numpy as np

def atoms2pygdata(atoms_in, ase=True):
    from sklearn.preprocessing import LabelEncoder
    import torch
    from torch_geometric.data import Data
    atoms_list = atoms_in if isinstance(atoms_in, (list, tuple)) else [atoms_in]
    le = LabelEncoder()
    g_list = []
    for atoms in atoms_list:
        x = le.fit_transform(atoms.numbers)
        if ase:
            g= Data(
                x       = torch.tensor(x,               dtype=torch.long),
                pos     = torch.tensor(atoms.positions, dtype=torch.float),
                cell    = np.array(atoms.cell),
                cell_inv= np.linalg.inv(np.array(atoms.cell)) if np.all(atoms.pbc) else None,
                pbc     = atoms.pbc,
                numbers = atoms.numbers,
            )
        else:
            g= Data(
                x    = torch.tensor(x,                    dtype=torch.long),
                pos  = torch.tensor(atoms.positions,      dtype=torch.float),
                box  = torch.tensor(np.array(atoms.cell), dtype=torch.float).sum(dim=0),
            )
        g_list.append(g)
    return g_list if isinstance(atoms_in, (list, tuple)) else g_list[0]


