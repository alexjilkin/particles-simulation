import numpy as np
from consts import W, H, SIGMA, EPSILON


def compute_total_energy(particles):
    kinetic = sum(0.5 * np.dot(p.vel, p.vel) for p in particles)
    potential = 0.0
    for i, pi in enumerate(particles):
        for j in range(i + 1, len(particles)):
            pj = particles[j]
            r_vec = pi.pos - pj.pos
            r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])
            r = np.linalg.norm(r_vec)

            if r < 3 * SIGMA and r > 1e-5:
                potential += 4 * EPSILON * ((SIGMA / r)**12 - (SIGMA / r)**6)
                
    return kinetic + potential

def is_far_enough(new_p, particles, min_dist):
    for p in particles:
        r_vec = new_p - p
        r_vec -= np.round(r_vec / np.array([W, H])) * np.array([W, H])  # periodic boundary
        if np.linalg.norm(r_vec) < min_dist:
            return False
    return True
