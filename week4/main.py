import copy
import numpy as np
import torch
from particles import Particle, create_lattice, lennard_jones_force, random_particles
from train import device, train_gnn
from graph_network import LJGnn
from consts import SIGMA, W, H
from simulate import simulate
import os
from draw import Drawable

Drawable.set_zoom(10)

n_particles = 225
N = 15000
epochs = 100

model_path = os.path.join("models", f"lj_net_{N}_{epochs}.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)


if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    with torch.serialization.safe_globals([LJGnn]):
        lj_net = LJGnn().to(device)
        lj_net.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Training model...")
    lj_net = train_gnn(n_particles, N, epochs)
    torch.save(lj_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

lj_net.eval()

# particles = random_particles(n_particles, 1.2 * SIGMA, 10 * SIGMA)

particles = create_lattice(2.1 * SIGMA, 10 , 20)

simulate(copy.deepcopy(particles), 0.005, True, lj_net)  
simulate(particles, 0.005, False)     
