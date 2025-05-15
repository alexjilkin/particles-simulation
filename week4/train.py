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

def particles_to_graph(positions, forces, cutoff=4 * SIGMA):
    num_particles = len(positions)
    edge_index = []
    edge_attr = []
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                rel_vector = positions[i] - positions[j]
                dist = np.linalg.norm(rel_vector)
                if dist < cutoff :  
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

from torch_geometric.data import Data

base_dir = os.path.dirname(__file__)  # folder of current script

def generate_data(N_uncorr, N_chains, chain_len, num_particles, dt=0.001):
    data_X, data_Y = [], []

    # --- Uncorrelated Samples ---
    for _ in tqdm(range(N_uncorr), desc="Generating uncorrelated samples"):
        particles = random_particles(num_particles, 1.15 * SIGMA, 5 * SIGMA)
        for p in particles:
            p.vel = np.random.randn(2) * 0.2
            p.acc = np.zeros(2)

        # One step of motion
        for p in particles:
            p.pos += p.vel * dt + 0.5 * p.acc * dt**2
            p.pos %= [W, H]

        forces = compute_forces(particles)
        positions = np.array([p.pos for p in particles])

        if np.all(np.abs(forces) < 100):
            data_X.append(positions)
            data_Y.append(forces)

    # --- Correlated Samples ---
    for _ in tqdm(range(N_chains), desc="Generating correlated chains"):
        particles = random_particles(num_particles, 1.15 * SIGMA, 5 * SIGMA)
        for p in particles:
            p.vel = np.random.randn(2) * 0.2
            p.acc = np.zeros(2)

        for _ in range(chain_len):
            old_accs = [p.acc.copy() for p in particles]
            forces = compute_forces(particles)

            for p, f in zip(particles, forces):
                p.force[:] = f

            for p, old_acc in zip(particles, old_accs):
                new_acc = p.force
                p.vel += 0.5 * (old_acc + new_acc) * dt
                p.acc = new_acc
                p.pos += p.vel * dt + 0.5 * p.acc * dt**2
                p.pos %= [W, H]

            positions = np.array([p.pos for p in particles])
            if np.all(np.abs(forces) < 100):
                data_X.append(positions)
                data_Y.append(forces)

    X = np.array(data_X)
    Y = np.array(data_Y)

    graph_dataset = []
    for pos, frc in zip(X, Y):
        graph = particles_to_graph(pos, frc)
        graph_dataset.append(graph)

    return X, Y, graph_dataset

def train_gnn(n_particles, N, epochs, lr=0.001):
    _, _, dataset = generate_data(N, 10, N // 10, n_particles)
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