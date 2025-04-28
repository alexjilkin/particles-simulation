import os

import torch

from utils import compute_total_energy, is_far_enough
from consts import W, H, SIGMA, EPSILON
from draw import Drawable
from network import LJNet

os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
import numpy as np

dt = 0.01
STEPS_PER_FRAME = 10



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

def lennard_jones_force_ml(lj_net: LJNet, p1: Particle, p2: Particle):
    r_vec = p1.pos - p2.pos
    r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])
    
    r = np.linalg.norm(r_vec)
    if r < 1e-5 or r > 2 * SIGMA:
        return np.zeros(2)

    with torch.no_grad():
        input_tensor = torch.tensor(r_vec, dtype=torch.float32)
        force = lj_net(input_tensor).numpy()
    return force

def random_particles(N, min_dist=0.6 * SIGMA):
    particles = []
    attempts = 0

    while len(particles) < N:
        new_pos = np.array([
            W/2 + (np.random.rand() - 0.5) * W/5, 
            H/2 + (np.random.rand() - 0.5) * H/5
        ])
        if is_far_enough(new_pos, [p.pos for p in particles], min_dist):
            p = Particle()
            p.pos = new_pos.copy()
            p.vel = np.zeros(2)
            particles.append(p)
            attempts += 1

    return particles

def create_lattice(spacing=1.1 * SIGMA, rows=3, cols=5):
    particles = []

    for i in range(rows):
        for j in range(cols):
            p = Particle()
            p.pos = np.array([W / 2 + j * spacing, H / 2 + i * spacing])
            p.vel = np.zeros(2) 
            particles.append(p)
    return particles

