import numpy as np
from torch import nn
import torch

"""
    deformation Network model classes.
    Both deformation Networks are based upon nerf MLP architectures. Separate Model is comprised of two separate MLP's.
    Connected model is one MLP. Both Networks take position and quaternion together with a time t. Networks calculate
    the deformation or change delta_x and delta_q s.t. the original positions have to be added. i.e. x(t) = x(0) + delta_x(t)
    and q(t) = delta_q(t) * q(0), as per coordinate and quaternion arithmetic. model calculates delta_x(t) and delta_q(t)
    It is assumed that both position and quaternions are normalized. Normalization of delta_q is built in. 
    TODO, check normalization for delta_x.
    
    since derivative for quaternion is quite simple (dq/dt = 1/2 w q), a simple second deformation network should be 
    sufficient to learn quaternion deformation. rotation does not necessarily depend on location, 
"""

# device = 'cuda:0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# coordinate_L = 10  #as per original nerf paper
coordinate_L = 10
quaternion_L = 10
pos_dim = 3
quat_dim = 4
layer_num = 4

class DeformationNetworkSeparate(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_x_1 = torch.nn.Linear(coordinate_L * 2 * pos_dim + 1, 256, device=device)
        self.linear_x_8 = torch.nn.Linear(256, 128, device=device)
        self.linear_x_9 = torch.nn.Linear(128, 3, device=device)

        self.linear_q_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + 1, 256, device=device)
        self.linear_q_8 = torch.nn.Linear(256, 128, device=device)
        self.linear_q_9 = torch.nn.Linear(128, 4, device=device)

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(256, 256, device=device),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.x_list = torch.nn.ModuleList([self.seq for i in range(layer_num)])

        self.q_list = torch.nn.ModuleList([self.seq for i in range(layer_num)])

        self.batch_norm_big = torch.nn.BatchNorm1d(256)
        self.batch_norm_small = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            quat_norm = torch.zeros(quat_dim).to(device)
            quat_norm[0] = 1
            return torch.zeros(pos_dim).to(device), quat_norm

        t = torch.tensor(t, dtype=torch.float).to(device)
        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_x = higher_x.clone().float().to(device).flatten(start_dim=1)
        input_x = torch.hstack((higher_x.clone().detach(), torch.ones(higher_x.shape[0])[None, :].T.to(device) * t))
        input_x = self.linear_x_1(input_x)
        input_x = self.batch_norm_big(input_x)
        input_x = self.relu(input_x)
        for i, l in enumerate(self.x_list):
            input_x = l(input_x)

        input_x = self.linear_x_8(input_x)
        input_x = self.batch_norm_small(input_x)
        input_x = self.relu(input_x)
        input_x = self.linear_x_9(input_x)

        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_q = higher_q.clone().float().to(device).flatten(start_dim=1)
        input_q = torch.hstack((higher_q.clone().detach(), torch.ones(higher_q.shape[0])[None, :].T.to(device) * t))
        input_q = self.linear_q_1(input_q)
        input_q = self.batch_norm_big(input_q)
        input_q = self.relu(input_q)
        for i, l in enumerate(self.q_list):
            input_q = l(input_q)
        input_q = self.linear_q_8(input_q)
        input_q = self.batch_norm_small(input_q)
        input_q = self.relu(input_q)
        input_q = self.linear_q_9(input_q)

        # normalization of q
        # input_q_2 = nn.functional.normalize(input_q, dim=0)

        return input_x, input_q


class DeformationNetworkCompletelyConnected(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + coordinate_L * 2 * pos_dim + 1, 512,
                                        device=device)
        self.linear_8 = torch.nn.Linear(512, 128, device=device)
        self.linear_9 = torch.nn.Linear(128, pos_dim + quat_dim, device=device)

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(512, 512, device=device),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )

        self.mod_list = torch.nn.ModuleList([self.seq for i in range(layer_num)])


        self.batch_norm_big = torch.nn.BatchNorm1d(512)
        self.batch_norm_small = torch.nn.BatchNorm1d(128)

        self.relu = torch.nn.ReLU()

    def forward(self, x, q, t):
        if t == 0:
            quat_norm = torch.zeros(quat_dim).to(device)
            quat_norm[0] = 1
            return torch.zeros(pos_dim).to(device), quat_norm

        higher_x = higher_dim_gamma(x, coordinate_L)
        higher_q = higher_dim_gamma(q, quaternion_L)
        higher_x = higher_x.clone().float().to(device).flatten(start_dim=1)
        higher_q = higher_q.clone().float().to(device).flatten(start_dim=1)
        input_total = torch.hstack((higher_x.clone().detach(), higher_q.clone().detach()))
        input_total = torch.hstack(
            (input_total.clone().detach(), torch.ones(input_total.shape[0])[None, :].T.to(device) * t))
        input_total = self.linear_1(input_total)
        input_total = self.batch_norm_big(input_total)
        input_total = self.relu(input_total)

        for i, l in enumerate(self.mod_list):
            input_total = l(input_total)

        input_total = self.linear_8(input_total)
        input_total = self.batch_norm_small(input_total)
        input_total = self.relu(input_total)
        input_total = self.linear_9(input_total)
        out = torch.split(input_total, pos_dim, dim=1)
        out_x = out[0]
        out_q = torch.hstack((out[1], out[2]))

        # normalization of quaternion
        # out_q_2 = nn.functional.normalize(out_q, dim=1)

        return out_x, out_q


class DeformationNetworkBilinearCombination(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_x_1 = torch.nn.Linear(coordinate_L * 2 * pos_dim + 1, 256, device=device)
        self.linear_x_8 = torch.nn.Linear(256, 128, device=device)

        self.linear_q_1 = torch.nn.Linear(quaternion_L * 2 * quat_dim + 1, 256, device=device)
        self.linear_q_8 = torch.nn.Linear(256, 128, device=device)

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(256, 256, device=device),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.batch_norm_big = torch.nn.BatchNorm1d(256)
        self.batch_norm_small = torch.nn.BatchNorm1d(128)

        self.x_list = torch.nn.ModuleList([self.seq for i in range(layer_num)])
        self.q_list = torch.nn.ModuleList([self.seq for i in range(layer_num)])

        self.relu = torch.nn.ReLU()

        self.bilin_end = torch.nn.Bilinear(128, 128, pos_dim + quat_dim, device=device)

    def forward(self, x, q, t):
        if t == 0:
            quat_norm = torch.zeros(quat_dim).to(device)
            quat_norm[0] = 1
            return torch.zeros(pos_dim).to(device), quat_norm

        t = torch.tensor(t, dtype=torch.float).to(device)
        input_x = higher_dim_gamma(x, coordinate_L)
        input_x = input_x.float().to(device).flatten(start_dim=1)
        input_x = torch.hstack((input_x.clone().detach(), torch.ones(input_x.shape[0])[None, :].T.to(device) * t))
        input_x = self.linear_x_1(input_x)
        input_x = self.batch_norm_big(input_x)
        input_x = self.relu(input_x)
        for i, l in enumerate(self.x_list):
            input_x = l(input_x)
        input_x = self.linear_x_8(input_x)
        input_x = self.batch_norm_small(input_x)
        input_x = self.relu(input_x)

        input_q = higher_dim_gamma(q, quaternion_L)
        input_q = input_q.float().to(device).flatten(start_dim=1)
        input_q = torch.hstack((input_q.clone().detach(), torch.ones(input_q.shape[0])[None, :].T.to(device) * t))
        input_q = self.linear_q_1(input_q)
        input_q = self.batch_norm_big(input_q)
        input_q = self.relu(input_q)
        for i, l in enumerate(self.q_list):
            input_q = l(input_q)
        input_q = self.linear_q_8(input_q)
        input_q = self.batch_norm_small(input_q)
        input_q = self.relu(input_q)

        input_total = self.bilin_end(input_x, input_q)

        out = torch.split(input_total, pos_dim, dim=1)
        out_x = out[0]
        out_q = torch.hstack((out[1], out[2]))

        # normalization of quaternion
        # out_q_2 = nn.functional.normalize(out_q, dim=1)

        return out_x, out_q


def higher_dim_gamma(p, length_ar):  # as per original NERF paper

    # tensor has to be torch, otherwise, it cant do gradients
    assert isinstance(p, torch.Tensor)

    # find dimension with 3 (N, 3)
    #expected shape (N, 3)
    if not len(p.shape) == 2:
        raise RuntimeError('shape in forward call needs to have 2 dimensions')
    if not (p.shape[1] == 3 or p.shape[1] == 4):
        raise RuntimeError('shape in forward call has wrong dimension. Shape should be (N, 3/4')
    # expand it  (N,3) -> (N, 3, exp)
    exp = torch.arange(length_ar).to(device)
    exp = (2 ** exp) * torch.pi
    gam_sin = torch.sin(torch.outer(p.flatten(), exp))
    gam_cos = torch.cos(torch.outer(p.flatten(), exp))

    expanded = torch.vstack((gam_sin.flatten(), gam_cos.flatten())).T.flatten().reshape(p.shape[0], p.shape[1],
                                                                                        2 * length_ar)

    return expanded
