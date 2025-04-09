import pygame
import numpy as np

W, H = 800, 800
N = 50
EPSILON = 10
SIGMA = 10     
DT = 1

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

class Particle:
    def __init__(self):
        self.pos = np.random.rand(2) * np.array([W, H])
        self.vel = np.random.randn(2) * 0.1
        self.force = np.zeros(2)

    def update(self):
        self.vel += self.force * DT
        self.pos += self.vel * DT
        self.pos %= [W, H]

    def draw(self, surf):
        pygame.draw.circle(surf, (255, 255, 255), self.pos.astype(int), 3)

def F(p1, p2):
    r_vec = p1.pos - p2.pos
    r = np.linalg.norm(r_vec)
    
    if r < 1e-5 or r > 3 * SIGMA:
        return np.zeros(2)
    
    factor = 24 * EPSILON * (2*(SIGMA/r)**12 - (SIGMA/r)**6) / r**2
    return factor * r_vec

particles = [Particle() for _ in range(N)]

running = True

while running:
    screen.fill((0, 0, 0))
    for p in particles:
        p.force[:] = 0.0

    for i, pi in enumerate(particles):
        for j in range(i + 1, N):
            pj = particles[j]
            f = F(pi, pj)
            pi.force -= f
            pj.force += f 

    for p in particles:
        p.update()
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
