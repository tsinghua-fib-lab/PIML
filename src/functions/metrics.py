# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('..')
import data.data as DATA


def collision_count(position, threshold, real_position=None, reduction=None):
    collisions = DATA.Pedestrians.collision_detection(position, threshold, real_position)
    if reduction == 'sum':
        out = torch.sum(collisions).item()
    elif reduction == 'mean':
        out = torch.mean(collisions).item()
    elif reduction is None:
        out = collisions
    else:
        raise NotImplementedError
    return out


def mae_with_time_mask(p, q, mask, reduction=None):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (t, N)
    """
    with torch.no_grad():
        mae = torch.norm(p[mask == 1] - q[mask == 1], p=2, dim=-1)
        if reduction == 'sum':
            out = torch.sum(mae).item()
        elif reduction == 'mean':
            out = torch.mean(mae).item()
    return out


def ot_with_time_mask(p, q, mask, eps=0.1, max_iter=100, reduction=None, dvs='cpu'):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (*c, t, N)
    """
    sinkhorn = SinkhornDistance(eps, max_iter, reduction, dvs=dvs)
    out = []
    for t in range(mask.shape[-2]):
        if torch.sum(mask[..., t, :]) > 1:
            ot, _, _ = sinkhorn(
                p[..., t, :, :][mask[..., t, :] == 1], q[..., t, :, :][mask[..., t, :] == 1])
            ot = ot.tolist()
            if type(ot) == list:
                out += ot
            else:
                out += [ot]
    if reduction == 'sum':
        out = np.sum(out)
    elif reduction == 'mean':
        out = np.mean(out)
    return out


def mmd_with_time_mask(p, q, mask, kernel_mul=2.0, kernel_num=5, fix_sigma=None, reduction=None):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (*c, t, N)
    """
    MMD = MaximumMeanDiscrepancy()
    if mask.dim() > 2:
        mask = mask.reshape(-1, mask.shape[-1])
        p = p.reshape(mask.shape[0], p.shape[-2], p.shape[-1])
        q = q.reshape(mask.shape[0], q.shape[-2], q.shape[-1])
    out = []
    for t in range(mask.shape[0]):
        if torch.sum(mask[t, :]) > 1:
            mmd = MMD(p[t, :, :][mask[t, :] == 1], q[t, :, :][mask[t, :] == 1])
            out.append(mmd.item())
    if reduction == 'sum':
        out = np.sum(out)
    elif reduction == 'mean':
        out = np.mean(out)
    return out


def wasserstein_distance_2d(distribution_p, distribution_q, eps=0.1, max_iter=100, reduction=None):
    sinkhorn = SinkhornDistance(eps, max_iter, reduction)
    dist, P, C = sinkhorn(distribution_p, distribution_q)
    return dist, P, C


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    MMD = MaximumMeanDiscrepancy()
    return MMD(source, target, kernel_mul, kernel_num, fix_sigma)


# Merged from https://github.com/dfdazac/wassdistance/blob/master/layers.py
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none', dvs='cpu'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dvs = dvs

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        mu = mu.to(self.dvs)
        nu = nu.to(self.dvs)
        u = u.to(self.dvs)
        v = v.to(self.dvs)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) -
                            torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * \
                (torch.log(nu + 1e-8) -
                 torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        # "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        # "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


# adapted from https://zhuanlan.zhihu.com/p/163839117
class MaximumMeanDiscrepancy(object):
    """docstring for MaximumMeanDiscrepancy"""

    def __init__(self):
        super(MaximumMeanDiscrepancy, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算Gram核矩阵
        source: sample_size_1 * feature_size 的数据
        target: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_ss矩阵，Source<->Source
        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
        # K_st矩阵，Source<->Target
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)

        # K_ts矩阵,Target<->Source
        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
        # K_tt矩阵,Target<->Target
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


if __name__ == "__main__":
    import numpy as np

    mmd = MaximumMeanDiscrepancy()
    # data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
    # data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(90,50)))
    data_1 = torch.zeros((3, 2))
    data_2 = torch.ones((3, 2))
    print("MMD Loss:", mmd(data_1, data_2))

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, dvs='cpu')
    data_1 = torch.zeros((3, 3, 2))
    data_2 = torch.ones((3, 3, 2))
    ot, _, _ = sinkhorn(data_1, data_2)
    ot = ot.tolist()
    print(type(ot))
    print('ot: ', ot)

    data_1 = torch.zeros((2, 3, 3, 2))
    data_2 = torch.ones((2, 3, 3, 2))
    mask = torch.ones((2, 3, 3))
    mmd = mmd_with_time_mask(data_1, data_2, mask)
    print(mmd)

    print(mae_with_time_mask(data_1, data_2, mask, reduction='mean'))
