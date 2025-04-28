import os

import torch

from utils import compute_total_energy
from consts import W, H, SIGMA, EPSILON
from draw import Drawable
from network import LJNet
from particles import Particle, create_lattice, lennard_jones_force, lennard_jones_force_ml, random_particles

os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np

STEPS_PER_FRAME = 1





def simulate(lj_net, particles, dt=0.01):
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

            for p in particles:
                p.force[:] = 0

            for i, pi in enumerate(particles):
                for j in range(i + 1, N):
                    pj = particles[j]
                    f = lennard_jones_force_ml(lj_net, pi, pj)
                    # f = lennard_jones_force(pi, pj)
                    pi.force -= f
                    pj.force += f

            for p, old_acc in zip(particles, old_accs):
                new_acc = p.force 
                p.vel += 0.5 * (old_acc + new_acc) * dt
                p.acc = new_acc

        screen.fill((0, 0, 0))
        for p in particles:
            p.draw(screen, 5)

        # energy = compute_total_energy(particles)
        # text = font.render(f"Total Energy: {energy:.2f}", True, (255, 255, 0))
        # screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
