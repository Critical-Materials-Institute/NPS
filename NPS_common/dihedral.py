
import numpy as np
import torch

def _dihedral(pos, pt=True):
    vecs = pos[:,1:]-pos[:,:-1]#;print(vecs.shape)
    if pt:
        norm = torch.linalg.norm
        cross = torch.cross
        # det = torch.linalg.det
        # sign = torch.sign
        if isinstance(vecs, np.ndarray): vecs = torch.from_numpy(vecs)
        vecs = vecs.double()
        # sgn = torch.sign(torch.linalg.det(vecs))
        sgn = (torch.linalg.det(vecs) >=0).int()*2-1
    else:
        norm = np.linalg.norm
        cross = np.cross
        # det = np.linalg.det
        # sign = np.sign
        sgn = np.sign(np.linalg.det(vecs))
    bond_len = norm(vecs, 2, 1, True)
    # print('bond len std/mean',np.std(bond_len), np.mean(bond_len));
    vecs = vecs/bond_len
    m = cross(vecs[:,0],vecs[:,1], -1);
    n = cross(vecs[:,1],vecs[:,2], -1);
    # angle123 = np.pi - np.arcsin(np.linalg.norm(m,axis=-1));
    # angle234 = np.pi - np.arcsin(np.linalg.norm(n,axis=-1));
    # print('123 bond angle', np.std(angle123), np.mean(angle123)); 
    # print('234 bond angle', np.std(angle234), np.mean(angle234)); 
    m= m/norm(m, 2, 1, True)
    n= n/norm(n, 2, 1, True)
    if pt:
        di = torch.arccos(torch.clip(torch.sum(m*n, 1), -1,1))
    else:
        di = np.arccos(np.clip(np.sum(m*n, 1), -1,1))
    di = di*(1+sgn)/2 + (2*np.pi-di)*(1-sgn)/2
    return di
    # print('dihedral angle', np.std(di), np.mean(di)); 
    # import matplotlib.pyplot as plt; 
    # plt.plot(np.arange(len(di)), di, marker='.'); plt.show()
    # np.save('dihedral.npy', np.stack((di[1:], di[1:]-di[:-1]),-1)[None,:].astype(np.float32))

def dihedral(pos, idx, vel=None, pt=True, unit='degree'):
    prefac = 180/np.pi if unit == 'degree' else 1
    d=pos[:, idx]#;print(d.shape)
    assert pt or (vel is None)
    if vel is None:
        return (_dihedral(d, pt=pt)*prefac,)
    else:
        assert pt
        if isinstance(d, np.ndarray): d = torch.from_numpy(d)
        d.requires_grad_(True)
        with torch.enable_grad():
            di = _dihedral(d, pt=pt)*prefac
            ddidpos = torch.autograd.grad(di.sum(), d)[0].detach()
            # print(ddidpos, ddidpos.shape, vel.shape, vel[:, idx].shape, (ddidpos*vel[:, idx]).shape)
            ddidt = (ddidpos*vel[:, idx]).sum((1,2))
        return di.detach().float(), ddidt.detach().float()

