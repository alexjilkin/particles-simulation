import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
g = 9.81
dt = 0.5

x0, y0 = 0.0, 200  
v0 = 10
angle_rad = np.deg2rad(np.pi / 3)
vx, vy = v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)

x_vals, y_vals = [x0], [y0]

x, y = x0, y0
while y >= 0:
    x += vx * dt
    y += vy * dt
    vy -= g * dt
    
    x_vals.append(x)
    y_vals.append(y)

plt.figure(figsize=(8, 5))
plt.scatter(x_vals, y_vals, label=f'dt = {dt}s')
plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('fig')