import os

import torch

from utils import compute_total_energy, is_far_enough
from consts import W, H, SIGMA, EPSILON
from draw import Drawable
from network import LJNet

os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np

dt = 0.005
STEPS_PER_FRAME = 20

class Particle(Drawable):
    def __init__(self):
        super().__init__(np.random.rand(2) * np.array([W, H]))
        self.vel = np.zeros(2)
        self.force = np.zeros(2)
        self.acc = np.zeros(2)

    def draw(self, surf, zoom):
        screen_pos = self.world_to_screen(zoom)
        pygame.draw.circle(surf, (255, 255, 255), screen_pos.astype(int), max(1, int(1 * zoom)))

def lennard_jones_force(p1, p2, max_force=50):
    r_vec = p1.pos - p2.pos
    r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])  
    r = np.linalg.norm(r_vec)

    if r < 1e-5 or r > 3 * SIGMA:
        return np.zeros(2)

    factor = 24 * EPSILON * (2 * (SIGMA / r)**12 - (SIGMA / r)**6) / r**2
    force = factor * r_vec

    mag = np.linalg.norm(force)
    if mag > max_force:
        force = force / mag * max_force

    return force

def random_particles(N, min_dist=1.1 * SIGMA, max_dist=1.5 * SIGMA):
    particles = []
    
    center = np.array([W/2, H/2])
    p0 = Particle()
    p0.pos = center.copy()
    p0.vel = np.zeros(2)
    particles.append(p0)

    while len(particles) < N:
        base_particle = np.random.choice(particles)
        
        offset = np.random.randn(2)
        offset = offset / np.linalg.norm(offset) 
        distance = np.random.uniform(min_dist, max_dist)
        new_pos = (base_particle.pos + offset * distance) % np.array([W, H])
        
        if is_far_enough(new_pos, [p.pos for p in particles], min_dist):
            p = Particle()
            p.pos = new_pos.copy()
            # p.vel = np.random.uniform(-1, 1, size=2)
            particles.append(p)

    return particles


def create_lattice(spacing=1.5 * SIGMA, rows=3, cols=5):
    particles = []

    for i in range(rows):
        for j in range(cols):
            p = Particle()
            p.pos = np.array([W / 2 + j * spacing, H / 2 + i * spacing])
            p.vel = np.zeros(2) 
            particles.append(p)
    return particles

