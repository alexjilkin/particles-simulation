import numpy as np
from consts import W, H, SIGMA, EPSILON

def compute_total_energy(particles):
    positions = np.array([p.pos for p in particles])
    velocities = np.array([p.vel for p in particles])
    
    # Kinetic energy: 0.5 * v^2 summed over all particles
    kinetic = 0.5 * np.sum(np.sum(velocities**2, axis=1))
    
    pos_i = positions[:, None, :]  # (N, 1, 2)
    pos_j = positions[None, :, :]  # (1, N, 2)
    r_vec = pos_i - pos_j
    r_vec -= np.round(r_vec / [W, H]) * [W, H]

    r = np.linalg.norm(r_vec, axis=-1)  # (N, N)
    mask = (r > 1e-5) & (r < 3 * SIGMA)
    
    # Avoid double-counting with upper triangle mask
    triu = np.triu(np.ones_like(r, dtype=bool), k=1)
    mask &= triu

    r_safe = np.where(mask, r, 1.0)  # prevent div by zero
    lj = np.zeros_like(r)
    lj[mask] = 4 * EPSILON * ((SIGMA / r_safe[mask])**12 - (SIGMA / r_safe[mask])**6)
    potential = np.sum(lj)

    return kinetic + potential

def is_far_enough(new_p, particles, min_dist):
    for p in particles:
        r_vec = new_p - p
        r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])  
        if np.linalg.norm(r_vec) < min_dist:
            return False
    return True
