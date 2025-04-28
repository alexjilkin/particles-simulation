import numpy as np
from particles import Particle, create_lattice, lennard_jones_force, random_particles

from train import train
from network import LJNet
from consts import W, H
from simulate import simulate

model = train(lennard_jones_force)

model_weights = model.state_dict()

lj_net = LJNet()
lj_net.load_state_dict(model_weights)
lj_net.eval()

# particles = random_particles(300)
particles = create_lattice()

dist = Particle()
dist.pos = np.array([W//2 + 3.0, H // 2 - 5.0])
dist.vel = np.array([0.0, 0.5])
particles.append(dist)

# simulate(lj_net, particles)