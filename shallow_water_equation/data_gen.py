import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  


nx, nt = 50, 100
dx, dt = 1.0/(nx-1), 5e-4
g, H = 9.8, 1.0 
energy_threshold = 0.1  

def generate_sequence(bs, nt):

    x = torch.linspace(0, 1, nx)
    eta = torch.stack([0.5 * torch.sin(0.3 * np.pi * x) + 0.7 * torch.sin(1 * np.pi * x) for _ in range(bs)])  # 水面位移
    u = torch.rand(bs, nx) * 0.1 
    eta_t = torch.zeros(bs, nx)  
    u_t = torch.zeros(bs, nx)

    def d2dx2(w):
        return (torch.roll(w,-1,dims=1) - 2*w + torch.roll(w,1,dims=1)) / dx**2

    data = []
    for n in range(nt):
     
        deta, du = d2dx2(eta), d2dx2(u)
        
        eta_half = eta + 0.5*dt*eta_t
        u_half = u + 0.5*dt*u_t
        
        deta_half, du_half = d2dx2(eta_half), d2dx2(u_half)
        
        eta_t = eta_t + dt*(- H*du_half)
        u_t = u_t + dt*(- g*deta_half) 
        
        eta = eta_half + 0.5*dt*eta_t
        u = u_half + 0.5*dt*u_t
        
        current_energy = compute_wave_energy(eta, u)

        while torch.any(torch.abs(current_energy - energy_threshold) > 1e-5):
            scale = (energy_threshold / current_energy).sqrt()  
            eta = eta * scale.unsqueeze(-1)
            u = u * scale.unsqueeze(-1)  
            current_energy = compute_wave_energy(eta, u)

        data.append(torch.stack([eta, u], dim=1).detach().numpy())

    return np.stack(data, axis=1)


def compute_wave_energy(eta, u):
    potential = 0.5 * g * eta.pow(2).sum(dim=1)
    kinetic = 0.5 * H * u.pow(2).sum(dim=1)
    total_energy = potential + kinetic
    return total_energy

lengths = [50, 100, 200, 300]  
bs = 128  
for length in lengths:
    sequence = generate_sequence(bs, length)
    np.save(os.path.join(current_directory, f'data/shallow_water_sequence_{length}.npy'), sequence)
    print(f'Sequence of length {length} with batch size {bs} saved as shallow_water_sequence_{length}.npy')


lengths = [50, 100, 200, 300]  
bs = 128  
for length in lengths:
    sequence = generate_sequence(bs, length)
    np.save(os.path.join(current_directory, f'data/shallow_water_sequence_{length}_test.npy'), sequence)
    print(f'Sequence of length {length} with batch size {bs} saved as shallow_water_sequence_{length}_test.npy')


sequence = np.load(os.path.join(current_directory, f'data/shallow_water_sequence_200.npy'))
print(sequence.shape)
sequence = sequence[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

x = np.linspace(0, 1, nx)
line1, = ax1.plot(x, sequence[0, 0], color='blue', linewidth=2)
line2, = ax2.plot(x, sequence[0, 1], color='red', linewidth=2)

ax1.set_ylabel('Surface Elevation')
ax1.set_ylim(-0.2, 0.2)
ax2.set_xlabel('x')
ax2.set_ylabel('Horizontal Velocity')
ax2.set_ylim(-0.2, 0.2)

def update(frame):
    eta, u = sequence[frame]
    line1.set_ydata(eta)
    line2.set_ydata(u)
    return line1, line2

ani = FuncAnimation(fig, update, frames=range(0, len(sequence), 2), interval=50, blit=True)
plt.tight_layout()
plt.show()