import torch
import torch.nn.functional as F
import numpy as np

class MLAPM:
    def __init__(self, **args):
        self.args = args
        return

    def step(self, position, velocity, desired_speed, destination, dt, radius=0.3):
        """
        position: [N, 2], 可以含有 nan
        velocity: [N, 2]
        desired_speed: [N, 1]
        destination: [N, 2]
        """

        force = torch.zeros_like(position)

        # acceleration to desired velocity
        ed = F.normalize(destination - position, dim=-1, p=2)
        force += (desired_speed * ed - velocity) / self.args['tau']

        # repulsive terms between pedestrians, consider field of view modulation
        vr = position.view(1, -1, 2) - position.view(-1, 1, 2)
        r = vr.norm(p=2, dim=2, keepdim=True)
        view = (torch.einsum('nk,nmk->nm', velocity, vr) > 0.).unsqueeze(-1)
        if self.args['version'] == 'raw':
            force -= (view * self.args['A'] * (self.args['B'] * r).exp() * F.normalize(vr, dim=-1, p=2)).sum(dim=1)
        elif self.args['version'] == 'GC':
            vv = velocity.view(1, -1, 2) - velocity.view(-1, 1, 2)
            cos = F.cosine_similarity(vr, vv, dim=-1).unsqueeze(-1)
            theta = -(vr[:, :, 0] * ed[:, None, 1] - vr[:, :, 1] * ed[:, None, 0]).sign() * self.args['theta'] / 180 * torch.pi # (N, N)
            theta.masked_fill_(theta == 0, self.args['theta'] / 180 * torch.pi)
            C = theta.cos()
            S = theta.sin()
            rotate_mat = torch.stack([C, -S, S, C], dim=-1).view(*theta.shape, 2, 2)
            direc = torch.einsum('NMij,NMj->NMi', rotate_mat, F.normalize(vr, dim=-1, p=2))
            force -= (view * self.args['A'] * (self.args['B'] * r + self.args['C'] * cos + self.args['D'] * r * cos).exp() * direc).sum(dim=1)
        elif self.args['version'] == 'UCY':
            vv = velocity.view(1, -1, 2) - velocity.view(-1, 1, 2)
            coll = (vr.norm(dim=-1) < radius * 2)
            coll |= (vr + vv * 1.0).norm(dim=-1) < radius * 2
            tmin = -(vr * vv).sum(dim=-1) / (vv * vv).sum(dim=-1)
            dmin = ((vr * vr).sum(dim=-1) - (vr * vv).sum(dim=-1) ** 2 / (vv * vv).sum(dim=-1)).sqrt()
            coll |= (tmin > 0) & (tmin < 1) & (dmin < radius * 2)
            theta = -(vr[:, :, 0] * ed[:, None, 1] - vr[:, :, 1] * ed[:, None, 0]).sign() * self.args['theta'] / 180 * torch.pi # (N, N)
            theta.masked_fill_(theta == 0, self.args['theta'] / 180 * torch.pi)
            C = theta.cos()
            S = theta.sin()
            rotate_mat = torch.stack([C, -S, S, C], dim=-1).view(*theta.shape, 2, 2)
            direc = torch.einsum('NMij,NMj->NMi', rotate_mat, F.normalize(vr, dim=-1, p=2))
            force -= (view * self.args['A'] * (self.args['B'] * r * coll + self.args['C'] * coll).exp() * direc).sum(dim=1)
        else:
            raise NotImplementedError

        action = velocity + force * dt
        return action

