import torch
import matplotlib.pyplot as plt
from models.mlapm import MLAPM

if __name__ == '__main__':
    N = 7
    dt = 0.08
    radius = 0.3
    theta = torch.linspace(0, 2 * torch.pi * (1 - 1. / N), N)
    position = torch.stack([10 * theta.cos(), 10 * theta.sin()], dim=-1).view(-1, 1, 2)
    velocity = torch.rand(N, 1, 2)
    mask = torch.full([N, 1, 1], True)
    desired_speed = torch.full([N, 2], 1.5)
    destination = -position.view(-1, 2)

    model = MLAPM(version='GC', tau=0.5, A=7.55, B=-3.00, C=0.2, D=-0.3, theta=56)

    for i in range(200):
        v = model.step(
            position[mask[:, -1, 0], -1, :], 
            velocity[mask[:, -1, 0], -1, :], 
            desired_speed[mask[:, -1, 0], :], 
            destination[mask[:, -1, 0], :], 
            dt=dt, radius=radius
        )
        p = position[mask[:, -1, 0], -1, :] + v * dt

        position = torch.concat([position, torch.full([N, 1, 2], float('nan'))], dim=1)
        velocity = torch.concat([velocity, torch.full([N, 1, 2], float('nan'))], dim=1)
        mask = torch.concat([mask, mask[:, (-1,), :]], dim=1)

        position[mask[:, -1, 0], -1, :] = p
        velocity[mask[:, -1, 0], -1, :] = v
        mask[:, -1, :] &= ~((position[:, -1, :] - destination).norm(dim=-1, keepdim=True) < radius)

        if not mask.any(): break

    plt.plot(position[:, :, 0].T, position[:, :, 1].T)
    plt.axis('equal')
    plt.show()
