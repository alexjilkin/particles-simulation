import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from particles import Particle, compute_forces, lennard_jones_force, random_particles, create_lattice
from network import LJNet
from tqdm import tqdm
from consts import W, H, SIGMA

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_network import LJGnn

def particles_to_graph(positions, forces, cutoff=5.0):
    num_particles = len(positions)
    edge_index = []
    edge_attr = []
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                rel_vector = positions[i] - positions[j]
                dist = np.linalg.norm(rel_vector)
                if dist < cutoff:  
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append([rel_vector[0], rel_vector[1], dist])
                    edge_attr.append([-rel_vector[0], -rel_vector[1], dist])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    x = torch.zeros((num_particles, 1))
    y = torch.tensor(forces, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_forces_old(particles):
    N = len(particles)
    forces = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = particles[i].pos - particles[j].pos
                r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])
                r = np.linalg.norm(r_vec)
                if 1 * SIGMA < r <= 5 * SIGMA:
                    forces[i] += lennard_jones_force(particles[i], particles[j])
    return forces

from torch_geometric.data import Data

base_dir = os.path.dirname(__file__)  # folder of current script

def generate_data(N_samples, num_particles, cutoff=3.0, dt=0.01):
    filename = os.path.join(base_dir, "data", f"data_{N_samples}.npz")

    if os.path.exists(filename):
        print(f"Loading data from {filename}")
        data = np.load(filename)
        X, Y = data['X'], data['Y']
    else:
        print(f"Generating {N_samples} samples and saving to {filename}")
        data_X, data_Y = [], []
        for _ in tqdm(range(N_samples), desc="Generating data"):
            particles = random_particles(num_particles, 1.15 * SIGMA, 5 * SIGMA)

            for p in particles:
                p.vel = np.random.randn(2) * 0.2
                p.acc = np.zeros(2)

            # Verlet-style position update
            for p in particles:
                p.pos += p.vel * dt + 0.5 * p.acc * dt**2
                p.pos %= [W, H] 
        
            forces = compute_forces_old(particles)
            positions = np.array([p.pos for p in particles])

            if np.any(forces != 0) and np.all(np.abs(forces) < 30):
                data_X.append(positions)
                data_Y.append(forces)

        X = np.array(data_X)
        Y = np.array(data_Y)

        # np.savez_compressed(filename, X=X, Y=Y)

    graph_dataset = []
    for i in range(len(X)):
        pos = X[i]
        frc = Y[i]

        graph = particles_to_graph(pos, frc, cutoff=cutoff)
        graph_dataset.append(graph)

    return X, Y, graph_dataset

def train_gnn(n_particles, N, epochs, lr=0.001):
    _, _, dataset = generate_data(N, n_particles)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LJGnn().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data.to(device)
            optimizer.zero_grad()
            
            pred_forces = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(pred_forces, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model