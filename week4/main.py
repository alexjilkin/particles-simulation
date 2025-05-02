import copy
import numpy as np
import torch
from particles import Particle, create_lattice, lennard_jones_force, random_particles

from train import device, train_gnn
from graph_network import LJGnn
from consts import SIGMA, W, H
from simulate import simulate

import os
N = 1000
epochs = 20

model_path = os.path.join("models", f"lj_net_{N}_{epochs}.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)


# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     with torch.serialization.safe_globals([LJGnn]):
#         lj_net = LJGnn().to(device)  
#         lj_net.load_state_dict(torch.load(model_path, map_location=device))
# else:
#     print("Training model...")
#     lj_net = train_gnn(N, epochs)
#     torch.save(lj_net.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

# lj_net.eval()

# particles = random_particles(30, 1.2 * SIGMA, 5 * SIGMA)

particles = create_lattice(2 * SIGMA, 5, 3)

# simulate(copy.deepcopy(particles), 0.01, True, lj_net)  
simulate(particles, 0.005, False)                        