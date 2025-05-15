import pygame
import numpy as np
import torch

from particles import STEPS_PER_FRAME, compute_forces, lennard_jones_force
from consts import EPSILON, SIGMA, W, H
from graph_network import LJGnn
from train import device
from torch_geometric.data import Data

from utils import compute_total_energy

def simulate(particles, dt=0.001, use_network=True, lj_net: LJGnn=0):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    positions = torch.tensor(np.array([p.pos for p in particles]), device=device, dtype=torch.float32)
    velocities = torch.tensor(np.array([p.vel for p in particles]), device=device, dtype=torch.float32)
    accelerations = torch.tensor(np.array([p.acc for p in particles]), device=device, dtype=torch.float32)

    running = True
    while running:
        for y in range(STEPS_PER_FRAME):
            positions += velocities * dt + 0.5 * accelerations * dt ** 2
            positions %= torch.tensor([W, H], device=device)

            old_accs = accelerations.clone()

            if use_network:
                rel_vecs = positions[:, None, :] - positions[None, :, :]
                rel_vecs -= torch.round(rel_vecs / torch.tensor([W, H], device=device)) * torch.tensor([W, H], device=device)
                dists = torch.norm(rel_vecs, dim=-1)

                mask = (dists < SIGMA * 4) & (dists > 1e-5)
                i_idx, j_idx = torch.nonzero(mask, as_tuple=True)

                edge_index = torch.stack([i_idx, j_idx])
                edge_attr = torch.stack([
                    rel_vecs[i_idx, j_idx, 0],
                    rel_vecs[i_idx, j_idx, 1],
                    dists[i_idx, j_idx]
                ], dim=1)

                lj_net.eval()
                with torch.no_grad():
                    accelerations = lj_net(positions, edge_index, edge_attr) 
            else:
                accelerations = lennard_jones_gpu(positions)
                
            velocities += 0.5 * (old_accs + accelerations) * dt
            
            if(y == 0):
                pos_cpu = positions.detach().cpu().numpy()
                
                for p, new_pos in zip(particles, pos_cpu):
                    p.pos[:] = new_pos                        

                screen.fill((0, 0, 0))
                for p in particles:
                    p.draw(screen)


        total_energy = compute_total_energy(particles)
        energy_text = font.render(f"Total Energy: {total_energy:.3f}", True, (255, 255, 255))
        screen.blit(energy_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

BOX  = torch.tensor([W, H], dtype=torch.float32, device=device)

def lennard_jones_gpu(pos, r_cut=5.0):
    rel = pos[:, None, :] - pos[None, :, :]
    rel -= torch.round(rel / BOX) * BOX                    
    r2  = (rel**2).sum(dim=-1)                             
    mask = (r2 > 1e-12) & (r2 < r_cut**2)
    inv_r6 = (SIGMA**2 / r2.clamp_min(1e-12))**3           # (σ/r)^6
    f_scalar = 24*EPSILON * (2*inv_r6**2 - inv_r6) / r2    # |F|/r
    f_scalar.masked_fill_(~mask, 0.0)
    forces = (f_scalar.unsqueeze(-1) * rel).sum(dim=1)     # sum_j F_ij
    return forces
