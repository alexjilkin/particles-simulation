# Existing imports & setup omitted here
import os
from typing import Callable

import numpy as np
from tqdm import tqdm

from consts import W, H, SIGMA, EPSILON
from particles import Particle, create_lattice, lennard_jones_force, random_particles

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import LJNet
from utils import is_far_enough


def train(force_func: Callable[[np.ndarray, np.ndarray], float], dt = 0.01, N=200):
    data_X = []
    data_Y = []

    for sim_step in tqdm(range(N), desc="Collecting samples"):
        particles = random_particles(np.random.randint(10, 100))
        N = len(particles)

        for i, pi in enumerate(particles):
            for j in range(i + 1, N):
                pj = particles[j]
                r_vec = pi.pos - pj.pos
                r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])

                r = np.linalg.norm(r_vec)
                if 1e-5 < r <= 2 * SIGMA:
                    force = force_func(pi, pj)
                    data_X.append(r_vec)
                    data_Y.append(force)

    particles = create_lattice(rows=5, cols=10)

    dist = Particle()
    dist.pos = np.array([W//2 + 3.0, H // 2 - 3])
    dist.vel = np.array([0.0, 0.5])

    particles.append(dist)
    N = len(particles)

    for i, pi in enumerate(particles):
            for j in range(i + 1, N):
                pj = particles[j]
                r_vec = pi.pos - pj.pos
                r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])

                r = np.linalg.norm(r_vec)
                if 1e-5 < r <= 2 * SIGMA:
                    force = force_func(pi, pj)
                    data_X.append(r_vec)
                    data_Y.append(force)

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)


    X = torch.tensor(data_X, dtype=torch.float32)
    Y = torch.tensor(data_Y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


    model = LJNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model
