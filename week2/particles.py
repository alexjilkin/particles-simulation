import os

from utils import compute_total_energy
from consts import W, H, SIGMA, EPSILON
from draw import Drawable

os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np

dt = 0.001
STEPS_PER_FRAME = 100

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

class Particle(Drawable):
    def __init__(self):
        super().__init__(np.random.rand(2) * np.array([W, H]))
        self.vel = np.random.randn(2) * 2
        self.force = np.zeros(2)
        self.acc = np.zeros(2)

    def draw(self, surf, zoom):
        screen_pos = self.world_to_screen(zoom)
        pygame.draw.circle(surf, (255, 255, 255), screen_pos.astype(int), max(1, int(1 * zoom)))

def lennard_jones_force(p1, p2, max_force=100):
    r_vec = p1.pos - p2.pos
    r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])  
    r = np.linalg.norm(r_vec)

    if r < 1e-5 or r > 2 * SIGMA:
        return np.zeros(2)

    factor = 24 * EPSILON * (2 * (SIGMA / r)**12 - (SIGMA / r)**6) / r**2
    force = factor * r_vec

    # mag = np.linalg.norm(force)
    # if mag > max_force:
    #     force = force / mag * max_force

    return force


def create_lattice(spacing=2 * SIGMA, rows=3, cols=8):
    particles = []

    for i in range(rows):
        for j in range(cols):
            p = Particle()
            p.pos = np.array([W / 2 + j * spacing, H / 2 + i * spacing])
            p.vel = np.zeros(2) 
            particles.append(p)
    return particles

particles = create_lattice()
dist = Particle()
dist.pos = np.array([W//2 + 3.0, H // 2 - 5.0])
dist.vel = np.array([0.0, 0.5])
particles.append(dist)

def simulate():
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
                    f = lennard_jones_force(pi, pj)
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
        clock.tick(600)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

simulate()