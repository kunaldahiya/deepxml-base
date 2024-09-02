import torch
import torch.nn.functional as F


def linear_3d(x, w, b=None):
    out = torch.matmul(x.unsqueeze(1), w.permute(0, 2, 1))
    if b is not None:
        out = out + b.permute(0, 2, 1)
    return out


def ip_sim_3d(x, w, b=None):
    return linear_3d(
        input=x, 
        weight=w, 
        bias=b if b is None else b.view(-1))


def cosine_sim_3d(x, w, *args):
    return linear_3d(
        input=F.normalize(x, dim=1), 
        weight=F.normalize(w, dim=1))


def ip_sim(x, w, b=None):
    return F.linear(
        input=x, 
        weight=w, 
        bias=b if b is None else b.view(-1))


def cosine_sim(x, w, *args):
    return F.linear(
        input=F.normalize(x, dim=1), 
        weight=F.normalize(w, dim=1))
