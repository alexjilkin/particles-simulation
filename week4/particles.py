import os

import torch

from utils import  is_far_enough
from consts import W, H, SIGMA, EPSILON
from draw import Drawable
from network import LJNet

os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np

dt = 0.005
STEPS_PER_FRAME = 10

class Particle(Drawable):
    def __init__(self):
        super().__init__(np.random.rand(2) * np.array([W, H]))
        self.vel = np.zeros(2)
        self.force = np.zeros(2)
        self.acc = np.zeros(2)

    def draw(self, surf):
        screen_pos = self.world_to_screen().astype(int)
        radius = max(1, int(1 * Drawable.zoom))
        pygame.draw.circle(surf, (255, 255, 255), screen_pos, radius, width=1)
        
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


def compute_forces(particles):
    positions = np.array([p.pos for p in particles])
    forces = np.zeros_like(positions)

    # Pairwise relative vectors with periodic boundary handling
    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    r_vec = pos_i - pos_j
    r_vec -= np.round(r_vec / [W, H]) * [W, H]

    r = np.linalg.norm(r_vec, axis=-1)
    mask = (r > 1e-5) & (r < 4 * SIGMA)
    r_safe = np.where(mask, r, 1.0)

    factor = np.zeros_like(r)
    factor[mask] = 24 * EPSILON * (
        2 * (SIGMA / r_safe[mask])**12 - (SIGMA / r_safe[mask])**6
    ) / r_safe[mask]**2

    f_vec = factor[..., None] * r_vec

    f_mag = np.linalg.norm(f_vec, axis=-1)
    clip_mask = f_mag > 50
    f_vec[clip_mask] *= (50 / f_mag[clip_mask])[..., None]

    forces += np.sum(f_vec, axis=1)
    forces -= np.sum(f_vec, axis=0)

    return forces
