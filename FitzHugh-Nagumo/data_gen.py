import torch
import numpy as np
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  

# 一维FitzHugh-Nagumo模型,使用Störmer-Verlet辛算法,加入能量约束


nx, nt = 25, 50 
dx, dt = 1e-0/(nx-1), 10e-4
c1, c2, a1, a2, b1, b2 = 1.0, 1.2, 5.05, 7.05, 0.01, 0.1

long_ts = 150
bs = 128

energy_threshold = 6.1


def generate_sequence(nt=nt):
    # 初始化u, v及其时间导数
    u = torch.rand(nx, requires_grad=False)
    v = torch.rand(nx, requires_grad=False)
    u_t = torch.zeros(nx)
    v_t = torch.zeros(nx)

    # 定义二阶导数算子(周期性边界条件)
    def d2dx2(w):
        return (torch.roll(w,-1) - 2*w + torch.roll(w,1)) / dx**2

    data = []
    for n in range(nt):
     
        du, dv = d2dx2(u), d2dx2(v)
        
        # 使用Störmer-Verlet辛算法更新u,v
        u_half = u + 0.5*dt*u_t
        v_half = v + 0.5*dt*v_t
        
        du_half, dv_half = d2dx2(u_half), d2dx2(v_half)
        
        u_t = u_t + dt*(c1**2*du_half - a1*u_half**3 + b1*u_half*v_half)
        v_t = v_t + dt*(c2**2*dv_half - a2*v_half**3 + b2*v_half*u_half)
        
        u = u_half + 0.5*dt*u_t
        v = v_half + 0.5*dt*v_t
        
        
        current_energy = compute_energy(u, v, du, dv)
        print(f"t={n*dt:.3f}, E={current_energy.item():.4f}")


        while current_energy <= energy_threshold:
            scale = (energy_threshold / current_energy).sqrt()
            u = u * scale
            v = v * scale
            # u_t = u_t * scale
            # v_t = v_t * scale
            current_energy = compute_energy(u, v, du, dv) + 1e-5
        print(f"t={n*dt:.3f}, E={current_energy.item():.4f}")

        data.append((u.detach().numpy(), v.detach().numpy()))

    return np.array(data)

# 定义计算能量的函数
def compute_energy(u, v, du, dv):
#     energy = 0.5 * (
#         (u**2).sum()*dx + (v**2).sum()*dx
#         + c1**2 * (du**2).sum()*dx + c2**2 * (dv**2).sum()*dx  
#         + a1 * (u**4).sum()*dx + a2 * (v**4).sum()*dx
#         - 2*b1 * (u**2*v).sum()*dx
#         - 2*b2 * (u*v**2).sum()*dx
#     )
    energy = 0.5 * ((u**2) + (v**2)).sum()
    return energy

def active_level(u, v):
    active_level = (u + v).sum()
    return active_level

# 生成bs个序列
data = []
for _ in range(bs):
    sequence = generate_sequence()
    data.append(sequence)
data = np.array(data)
print(f"Generated dataset of shape {data.shape}")
np.save(current_directory + "/data/data.npy", data)

test_data = []
for _ in range(bs):  
    sequence = generate_sequence()
    test_data.append(sequence)
test_data = np.array(test_data)
print(f"Generated dataset of shape {test_data.shape}")  
np.save(current_directory + "/data/test_data.npy", test_data)



long_test_data = []
for _ in range(bs):  
    sequence = generate_sequence(long_ts)
    long_test_data.append(sequence)
long_test_data = np.array(long_test_data)
print(f"Generated dataset of shape {long_test_data.shape}")  
np.save(current_directory + "/data/long_test_data.npy", long_test_data)

# print(energy_threshold / dx)