import torch
import numpy as np
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  

nx, nt = 25, 50 
dx, dt = 1e-0/(nx-1), 5e-4
alpha, beta, gamma, delta = 1.1, 0.4, 0.1, 0.4

long_ts = 150
bs = 128

population_threshold = 75.0


def generate_sequence(nt=nt):

    u = torch.rand(nx, requires_grad=False)
    v = torch.rand(nx, requires_grad=False)
    u_t = torch.zeros(nx)  
    v_t = torch.zeros(nx)


    def d2dx2(w):
        return (torch.roll(w,-1) - 2*w + torch.roll(w,1)) / dx**2

    data = []
    for n in range(nt):
     
        du, dv = d2dx2(u), d2dx2(v)

        u_half = u + 0.5*dt*u_t
        v_half = v + 0.5*dt*v_t
        
        du_half, dv_half = d2dx2(u_half), d2dx2(v_half)
        
        u_t = u_t + dt*(alpha*u_half - beta*u_half*v_half + du_half)  
        v_t = v_t + dt*(delta*u_half*v_half - gamma*v_half + dv_half)
        
        u = u_half + 0.5*dt*u_t
        v = v_half + 0.5*dt*v_t
        
        
        current_population = compute_population(u, v)
        print(f"t={n*dt:.3f}, Population={current_population.item():.4f}")


        while np.abs(current_population - population_threshold) > 1e-4:
            scale = (population_threshold / current_population).sqrt()  
            u = u * scale
            v = v * scale
            current_population = compute_population(u, v)
        print(f"t={n*dt:.3f}, Population={current_population.item():.4f}")  

        data.append((u.detach().numpy(), v.detach().numpy()))

    return np.array(data)


def compute_population(u, v):
    population = (u + v).sum()
    return population



data = []
for _ in range(bs):
    sequence = generate_sequence()
    data.append(sequence)
data = np.array(data)
print(f"Generated dataset of shape {data.shape}")
np.save(current_directory + "/data/LK_data.npy", data)

test_data = []  
for _ in range(bs):
    sequence = generate_sequence()
    test_data.append(sequence)
test_data = np.array(test_data)
print(f"Generated dataset of shape {test_data.shape}")
np.save(current_directory + "/data/LK_test_data.npy", test_data)


long_test_data = []
for _ in range(bs):
    sequence = generate_sequence(long_ts) 
    long_test_data.append(sequence)
long_test_data = np.array(long_test_data)
print(f"Generated dataset of shape {long_test_data.shape}")
np.save(current_directory + "/data/LK_long_test_data.npy", long_test_data)