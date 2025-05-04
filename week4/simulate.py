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

    N = len(particles)
    running = True
    while running:
        for _ in range(STEPS_PER_FRAME):
            for p in particles:
                p.pos += p.vel * dt + 0.5 * p.acc * dt**2
                p.pos %= [W, H]

            old_accs = [p.acc.copy() for p in particles]

            if use_network:
                positions = np.array([p.pos for p in particles])
                N = len(particles)

                edge_index = []
                edge_attr = []
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            vec = positions[i] - positions[j]
                            dist = np.linalg.norm(vec)
                            if dist < 5:
                                edge_index.append([i, j])
                                edge_attr.append([vec[0], vec[1], dist])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32) 
                
                x = torch.tensor(positions, dtype=torch.float32)
                graph = Data(x=x, edge_index=edge_index.to(device), edge_attr=edge_attr)

                lj_net.eval()
                with torch.no_grad():
                    pred_forces = lj_net(graph.x.to(device), graph.edge_index.to(device), graph.edge_attr.to(device))

                forces = pred_forces.cpu().numpy()
            else:
                forces = compute_forces(particles)
                
            for p, f in zip(particles, forces):
                p.force[:] = f
                    
            for p, old_acc in zip(particles, old_accs):
                new_acc = p.force
                p.vel += 0.5 * (old_acc + new_acc) * dt
                p.acc = new_acc

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


