
import numpy as np
import torch

def distance(pos, idx, vel=None, pt=True):
    x = pos[:, idx]
    v = vel[:, idx] if vel is not None else None
    assert pt or (vel is None)
    assert pt
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(v, np.ndarray): v = torch.from_numpy(v)
    if vel is None:
        return (torch.norm(x[:,0] - x[:,1], dim=1)[..., None],)
    else:
        x.requires_grad_(True)
        with torch.enable_grad():
            cv = torch.norm(x[:,0] - x[:,1], dim=1)
            dcvdx = torch.autograd.grad(cv.sum(), x)[0].detach()
            # print(ddidpos, ddidpos.shape, vel.shape, vel[:, idx].shape, (ddidpos*vel[:, idx]).shape)
            dcvdt = (dcvdx*v).sum((1,2))
        return cv.detach().float(), dcvdt.detach().float()

def ahelix_content(pos, idx, vel=None, pt=True):
    x = pos[:, idx]
    v = vel[:, idx] if vel is not None else None
    assert pt or (vel is None)
    assert pt
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(v, np.ndarray): v = torch.from_numpy(v)
    if vel is None:
        return (torch.norm(x[:,0] - x[:,1], dim=1)[..., None],)
    else:
        x.requires_grad_(True)
        with torch.enable_grad():
            cv = torch.norm(x[:,0] - x[:,1], dim=1)
            dcvdx = torch.autograd.grad(cv.sum(), x)[0].detach()
            # print(ddidpos, ddidpos.shape, vel.shape, vel[:, idx].shape, (ddidpos*vel[:, idx]).shape)
            dcvdt = (dcvdx*v).sum((1,2))
        return cv.detach().float(), dcvdt.detach().float()

