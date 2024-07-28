import torch
import torch.nn as nn
import time
# def euler_ode_solver(func, y0, t, u=None):
#     dt = t[1] - t[0]
#     y = y0
#     ys = [y0]
#     if u is None:
#         u = torch.zeros_like(y0)

#     for i in range(len(t) - 1):
#         t_start, t_end = t[i], t[i+1]
#         y = y + (func(t_start, y) - u) * dt
#         t_start += dt
#         ys.append(y)
#     return torch.stack(ys) 




def euler_ode_solver(func, y0, t, g=None, h=None):
    dt = t[1] - t[0]
    y = y0
    ys = [y0]
    v = torch.zeros_like(y0).to(y0.device)
    if h is not None:
        v = h(y0)
    for i in range(len(t) - 1):
        u = torch.zeros_like(y0).to(y0.device)
        if g is not None:
            u = g(y)
        t_start, t_end = t[i], t[i+1]
        y = y + (func(t_start, y) - u + v) * dt
        t_start += dt
        ys.append(y)
    return torch.stack(ys) 



def euler_ode_solver_with_sac(func, y0, t, g=None, h=None):
    dt = t[1] - t[0]
    y = y0
    ys = [y0]
    for i in range(len(t) - 1):
        u = torch.zeros_like(y0).to(y0.device)
        if g is not None:
            u, _, _ = g(y)
        t_start, t_end = t[i], t[i+1]
        y = y + (func(t_start, y) - u) * dt
        t_start += dt
        ys.append(y)
    return torch.stack(ys)




def euler_continuous_ode_solver(func, y0, t, dt=0.1, g=None):
    interval = t[1] - t[0]
    y = y0
    ys = [y0]

    len_t_span = int(1/dt)
    for i in range(len(t) - 1):
        t_start, t_end = t[i], t[i+1]
        for j in range(len_t_span):
            u = torch.zeros_like(y0).to(y0.device)
            if g is not None:
                u, _, _ = g(y)
            y = y + (func(t_start, y) - u) * interval * dt
        t_start += interval
        ys.append(y)
    return torch.stack(ys)



class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        y = self.net(y)
        return y


class SNDE(nn.Module):
    def __init__(self, f_model, constraint_fn, gamma):
        super(SNDE, self).__init__()
        self.func = SNDEDynamic(f_model, constraint_fn, gamma)
        self.loss_fn = nn.MSELoss()

    def forward(self, y0, t):
        out = euler_ode_solver(self.func, y0, t)
        return out
    
    def criterion(self, prediction, target):
        return self.loss_fn(prediction, target)


class SNDEDynamic(nn.Module):
    def __init__(self, f_model, g_fn, gamma):
        super(SNDEDynamic, self).__init__()
        self.f_model = f_model
        self.g_fn = g_fn
        self.gamma = gamma

    # def forward(self, t, u):
    #     f_u = self.f_model(u)
    #     g_u = self.g_fn(u)
    #     G_u = self.compute_jacobian(u)
    #     F_u = self.compute_pseudo_inverse(G_u)
    #     stable_term = self.gamma * torch.bmm(F_u, g_u.unsqueeze(-1)).squeeze(-1)
    #     du_dt = f_u - stable_term
    #     return du_dt
    
    def forward(self, t, u):
        f_u = self.f_model(t, u)
        g_u = self.g_fn(u)
        G_u = self.compute_jacobian(u)
        F_u = self.compute_pseudo_inverse(G_u)
        stable_term = self.gamma * torch.bmm(F_u, g_u.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        du_dt = f_u - stable_term
        return du_dt

    # def compute_jacobian(self, u):
    #     bs = u.shape[0]
    #     u.requires_grad_(True)
    #     g_u = self.g_fn(u)
    #     G_u = torch.zeros(bs, constraint_dim, input_dim).to(u.device)
    #     for i in range(constraint_dim):
    #         grad_outputs = torch.zeros_like(g_u)
    #         grad_outputs[:, i] = 1
    #         G_u[:, i, :] = torch.autograd.grad(g_u, u, grad_outputs=grad_outputs, create_graph=True)[0]
    #     return G_u

    def compute_jacobian(self, u):
        bs = u.shape[0]
        input_dim = u.shape[-1]
        u = u.clone().detach().requires_grad_(True)
        # u.requires_grad_(True)
        g_u = self.g_fn(u)
        G_u = torch.zeros(bs, input_dim).to(u.device)
        grad_outputs = torch.ones_like(g_u).to(u.device)
        G_u[:, :] = torch.autograd.grad(g_u, u, grad_outputs=grad_outputs, create_graph=True)[0]
        G_u =  G_u.unsqueeze(1)
        return G_u
    
    def compute_pseudo_inverse(self, G_u):
        G_u_t = torch.transpose(G_u, -2, -1)
        G_u_pinv = torch.pinverse(torch.bmm(G_u, G_u_t))
        F_u = torch.bmm(G_u_t, G_u_pinv)
        return F_u



class SAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SAC, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, y):
        mean = self.actor_mean(y)
        log_std = self.actor_log_std(y)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, self.critic(y)


class NeuralODE_SAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, sac_hidden_dim=64, sac_num_layers=2):
        super(NeuralODE_SAC, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = SAC(input_dim, sac_hidden_dim, sac_num_layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, y0, t):
        out = euler_ode_solver_with_sac(self.func, y0, t, self.g_func)
        return out
    
    def criterion(self, prediction, target, constrain_loss_func=None):
        log_probs = []
        values = []

        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        if constrain_loss_func is not None:
            constrain_loss = constrain_loss_func(prediction)
        reward = -loss + (-constrain_loss * 0.1)

        for i in range(prediction.shape[1]):
            state = prediction[:, i, ...]
            _, log_prob, value = self.g_func(state)
            log_probs.append(log_prob)
            values.append(value)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        q_values = reward * 0.5

        tau =  0.1
        log_alpha  = -3

        soft_q_values = (1 - tau) * q_values + tau * values
        value_loss = 0.5 * torch.mean((values - soft_q_values) ** 2)
        log_alpha_tensor = torch.tensor(log_alpha, dtype=torch.float32)
        advantages = q_values - values.detach()
        policy_loss = torch.mean(torch.exp(log_alpha_tensor) * log_probs - advantages)
        entropy_regularization = -torch.mean(torch.exp(log_alpha_tensor) * log_probs)
        sac_loss = value_loss + policy_loss + entropy_regularization
    
        sac_loss *= 0.025
        return sac_loss + loss





class NeuralODE_SAC_CONTINUOUS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, sac_hidden_dim=64, sac_num_layers=2):
        super(NeuralODE_SAC_CONTINUOUS, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = SAC(input_dim, sac_hidden_dim, sac_num_layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, y0, t):
        out = euler_continuous_ode_solver(self.func, y0, t, 0.2, self.g_func)        
        return out
    
    def criterion(self, prediction, target, constrain_loss_func=None):
        log_probs = []
        values = []

        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        if constrain_loss_func is not None:
            constrain_loss = constrain_loss_func(prediction)
        reward = -loss + (-constrain_loss * 0.1)

        for i in range(prediction.shape[1]):
            state = prediction[:, i, ...]
            _, log_prob, value = self.g_func(state)
            log_probs.append(log_prob)
            values.append(value)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        q_values = reward * 0.1

        tau =  0.1
        log_alpha  = -3

        soft_q_values = (1 - tau) * q_values + tau * values
        value_loss = 0.5 * torch.mean((values - soft_q_values) ** 2)
        log_alpha_tensor = torch.tensor(log_alpha, dtype=torch.float32)
        advantages = q_values - values.detach()
        policy_loss = torch.mean(torch.exp(log_alpha_tensor) * log_probs - advantages)
        entropy_regularization = -torch.mean(torch.exp(log_alpha_tensor) * log_probs)
        sac_loss = value_loss + policy_loss + entropy_regularization
    
        sac_loss *= 0.01
        return sac_loss + loss










class NeuralODE_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, hidden_dim_mlp=128, num_layers_mlp=4):
        super(NeuralODE_MLP, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = MLP(input_dim, hidden_dim_mlp, num_layers_mlp)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, y0, t):
        out = euler_ode_solver(self.func, y0, t, self.g_func)
        return out
    
    def criterion(self, prediction, target, constrain_loss_func=None):
        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        if constrain_loss_func is not None:
            constrain_loss = constrain_loss_func(prediction)
        return loss + constrain_loss

    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()  
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim))     
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        y = self.net(y)
        return y



class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, y0, t):
        out = euler_ode_solver(self.func, y0, t)
        out = out
        return out
    
    def criterion(self, prediction, target, constrain_loss_func=None):
        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        if constrain_loss_func is not None:
            constrain_loss = constrain_loss_func(prediction)
        return loss + constrain_loss




class SAC_Manifold(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SAC_Manifold, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, y):
        mean = self.actor_mean(y)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean).to(y.device), mean)  # 将 NaN 替换为 0
        mean = torch.where(torch.isinf(mean), torch.zeros_like(mean).to(y.device), mean)  # 将 Infinity 替换为 0

        log_std = self.actor_log_std(y)
        # log_std = torch.clamp(log_std, min=1e-6, max=10)
        log_std = torch.where(torch.isnan(log_std), torch.zeros_like(log_std).to(y.device), log_std)  # 将 NaN 替换为 0
        log_std = torch.where(torch.isinf(log_std), torch.zeros_like(log_std).to(y.device), log_std)  # 将 Infinity 替换为 0
        
        std = torch.nn.functional.softplus(log_std)


        assert torch.all(std > 0), "Standard deviations must be positive"
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, self.critic(y)



class ManifoldDynamic(nn.Module):
    def __init__(self, f_model, g_fn, k_func, alpha=6e1, beta=2e1, sigma=5e-0):
        super(ManifoldDynamic, self).__init__()
        self.f_model = f_model
        self.g_fn = g_fn
        self.k_func = k_func
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    
    def forward(self, t, x):
        bs, n = x.shape
        f_out = self.f_model(t, x)
        k_out = self.k_func(x)
        G_k_out = self.g_fn(x)[0]

        dx_dt = f_out + G_k_out
        J_x = self.compute_k_jacobian(t, x)  # jacobian_state 的形状为 (bs, n, n)

        k_out_norm = torch.norm(k_out, dim=1)
        J_x_f_out_norm = torch.norm(torch.bmm(J_x, f_out.unsqueeze(-1)).squeeze(-1), dim=1)

        constraint_1 = J_x_f_out_norm - self.alpha * k_out_norm ** (2 * self.sigma - 1)
        constraint_2 = torch.sum(k_out * torch.bmm(J_x, G_k_out.unsqueeze(-1)).squeeze(-1), dim=1) - self.beta * k_out_norm ** (2 * self.sigma)

        epsilon = 1e-8
        constraint_1_mask = constraint_1 > epsilon
        constraint_2_mask = constraint_2 < -epsilon

        # dx_dt_updated = dx_dt.clone()
        # dx_dt_updated[constraint_1_mask | constraint_2_mask] = 0

        dx_dt_updated = torch.where(constraint_1_mask.unsqueeze(-1) | constraint_2_mask.unsqueeze(-1), torch.zeros_like(dx_dt), dx_dt)
        dx_dt_updated = (dx_dt_updated + dx_dt) / 2
        return dx_dt_updated


    def compute_k_jacobian1(self, u):
        bs = u.shape[0]
        input_dim = u.shape[-1]
        u = u.clone().detach().requires_grad_(True)
        # u.requires_grad_(True)
        g_u = self.k_func(u)
        G_u = torch.zeros(bs, input_dim).to(u.device)
        grad_outputs = torch.ones_like(g_u).to(u.device)
        G_u[:, :] = torch.autograd.grad(g_u, u, grad_outputs=grad_outputs, create_graph=True)[0]
        G_u =  G_u.unsqueeze(1)
        return G_u
    
    def compute_k_jacobian(self, t, u, num_random_vectors=0):
        bs, n = u.shape
        u = u.clone().detach().requires_grad_(True)  # 确保 u 允许梯度计算
        jacobian_state = torch.zeros(bs, n, n).to(u.device)
        
        if num_random_vectors >= 1:
            eps = 1e-8
            num_random_vectors = 10
            bs, n = u.shape
            jacobian_est = torch.zeros(bs, n, n).to(u.device)
            random_vectors = torch.randn(num_random_vectors, n).to(u.device)
            for j in range(num_random_vectors):
                v = random_vectors[j]
                # Perturb u along the dimension of v
                u_perturbed_plus = u + eps * v
                u_perturbed_minus = u - eps * v

                # Apply g_fn and handle tuple output
                k_plus_result = self.k_func(u_perturbed_plus)
                k_minus_result = self.k_func(u_perturbed_minus)

                # Extract the first component if the output is a tuple
                k_plus = k_plus_result
                k_minus =  k_minus_result

                # Calculate the directional derivative
                directional_derivative = (k_plus - k_minus) / (2 * eps)
                # If the output is not two-dimensional, we need to adjust
                if directional_derivative.dim() == 1:
                    directional_derivative = directional_derivative.unsqueeze(-1)  # Add a singleton dimension for compatibility
                # Ensure v is two-dimensional (add batch dimension)
                v = v.unsqueeze(0).expand(bs, -1)
                jacobian_est += torch.einsum('bi,bj->bij', directional_derivative, v)
            jacobian_est /= num_random_vectors
            jacobian_state =  jacobian_est
        else:
            k_u = self.k_func(u)
            for i in range(n):
                grad_outputs = torch.zeros_like(k_u).to(u.device)
                grad_outputs[:, i] = 1
                jacobian_state[:, :, i] = torch.autograd.grad(k_u, u, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        return jacobian_state
 

class NeuralODE_SAC_Manifold(nn.Module):
    def __init__(self, distance_to_manifold_fn, input_dim, hidden_dim, num_layers, sac_hidden_dim=64, sac_num_layers=2):
        super(NeuralODE_SAC_Manifold, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = SAC_Manifold(input_dim, sac_hidden_dim, sac_num_layers)
        self.loss_fn = nn.MSELoss()
        self.k_func = distance_to_manifold_fn
        self.dynamic = ManifoldDynamic(self.func, self.g_func, self.k_func)

    def forward(self, y0, t):

        out = euler_ode_solver(self.dynamic, y0, t)
        return out
        
    def criterion(self, prediction, target, constrain_loss_func=None):
        log_probs = []
        values = []

        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        if constrain_loss_func is not None:
            constrain_loss = constrain_loss_func(prediction)
        reward = -loss + (-constrain_loss * 0.1)

        for i in range(prediction.shape[1]):
            state = prediction[:, i, ...]
            _, log_prob, value = self.g_func(state)
            log_probs.append(log_prob)
            values.append(value)

        log_probs = torch.cat(log_probs, dim=0)
        values = torch.cat(values, dim=0)
        q_values = reward

        tau =  0.1
        log_alpha  = -3

        soft_q_values = (1 - tau) * q_values + tau * values
        value_loss = 0.5 * torch.mean((values - soft_q_values) ** 2)
        log_alpha_tensor = torch.tensor(log_alpha, dtype=torch.float32)
        advantages = q_values - values.detach()
        policy_loss = torch.mean(torch.exp(log_alpha_tensor) * log_probs - advantages)
        entropy_regularization = -torch.mean(torch.exp(log_alpha_tensor) * log_probs)
        sac_loss = value_loss + policy_loss + entropy_regularization
    
        sac_loss *= 0.025
        return sac_loss + loss
    










class NeuralODE_SAC_ManifoldLoss(nn.Module):
    def __init__(self, distance_to_manifold_fn, input_dim, hidden_dim, num_layers, sac_hidden_dim=64, sac_num_layers=2):
        super(NeuralODE_SAC_ManifoldLoss, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = SAC_Manifold(input_dim, sac_hidden_dim, sac_num_layers)
        self.loss_fn = nn.MSELoss()
        self.k_func = distance_to_manifold_fn

    def forward(self, y0, t):
        out = euler_ode_solver_with_sac(self.func, y0, t, self.g_func)
        return out
    
    def manifold_constraint_loss(self, x):

        self.alpha = 1e-5
        self.beta = 2e-5
        self.sigma = 5e-5

        f_out = self.func(0, x)
        k_out = self.k_func(x)
        G_k_out = self.g_func(x)[0]

        J_x = self.compute_k_jacobian(x, 10)  # jacobian_state 的形状为 (bs, n, n)

        k_out_norm = torch.norm(k_out, dim=1)
        J_x_f_out_norm = torch.norm(torch.bmm(J_x, f_out.unsqueeze(-1)).squeeze(-1), dim=1)

        constraint_1 = J_x_f_out_norm - self.alpha * k_out_norm ** (2 * self.sigma - 1)
        constraint_2 = torch.sum(k_out * torch.bmm(J_x, G_k_out.unsqueeze(-1)).squeeze(-1), dim=1) - self.beta * k_out_norm ** (2 * self.sigma)

        loss_constraint_1 = torch.mean(torch.relu(constraint_1))
        loss_constraint_2 = torch.mean(torch.relu(-constraint_2))

        loss_constraint_1 = loss_constraint_1 * 1e-3
        loss_constraint_2 = loss_constraint_2 * 1e-3 

        loss = loss_constraint_1 + loss_constraint_2
        # print(loss_constraint_1, loss_constraint_2)
        return loss
    



    def compute_k_jacobian(self, u, num_random_vectors=0):
        bs, n = u.shape
        u = u.clone().detach().requires_grad_(True)  # 确保 u 允许梯度计算
        jacobian_state = torch.zeros(bs, n, n).to(u.device)
        
        if num_random_vectors >= 1:
            eps = 1e-8
            num_random_vectors = 10
            bs, n = u.shape
            jacobian_est = torch.zeros(bs, n, n).to(u.device)
            random_vectors = torch.randn(num_random_vectors, n).to(u.device)
            for j in range(num_random_vectors):
                v = random_vectors[j]
                # Perturb u along the dimension of v
                u_perturbed_plus = u + eps * v
                u_perturbed_minus = u - eps * v

                # Apply g_fn and handle tuple output
                k_plus_result = self.k_func(u_perturbed_plus)
                k_minus_result = self.k_func(u_perturbed_minus)

                # Extract the first component if the output is a tuple
                k_plus = k_plus_result
                k_minus =  k_minus_result

                # Calculate the directional derivative
                directional_derivative = (k_plus - k_minus) / (2 * eps)
                # If the output is not two-dimensional, we need to adjust
                if directional_derivative.dim() == 1:
                    directional_derivative = directional_derivative.unsqueeze(-1)  # Add a singleton dimension for compatibility
                # Ensure v is two-dimensional (add batch dimension)
                v = v.unsqueeze(0).expand(bs, -1)
                jacobian_est += torch.einsum('bi,bj->bij', directional_derivative, v)
            jacobian_est /= num_random_vectors
            jacobian_state =  jacobian_est
        else:
            k_u = self.k_func(u)
            for i in range(n):
                grad_outputs = torch.zeros_like(k_u).to(u.device)
                grad_outputs[:, i] = 1
                jacobian_state[:, :, i] = torch.autograd.grad(k_u, u, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        return jacobian_state
                
        
    def criterion(self, prediction, target, constrain_loss_func=None):
        log_probs = []
        values = []

        loss = self.loss_fn(prediction, target)
        constrain_loss = 0.0
        

        for i in range(prediction.shape[1]):
            state = prediction[:, i, ...]
            _, log_prob, value = self.g_func(state)
            log_probs.append(log_prob)
            values.append(value)
            if i == prediction.shape[1]-1:
                constrain_loss += self.manifold_constraint_loss(state)

        reward = -loss + (-constrain_loss * 1e-2)

        log_probs = torch.cat(log_probs, dim=0)
        values = torch.cat(values, dim=0)
        q_values = reward

        tau =  0.1
        log_alpha  = -3

        soft_q_values = (1 - tau) * q_values + tau * values
        value_loss = 0.5 * torch.mean((values - soft_q_values) ** 2)
        log_alpha_tensor = torch.tensor(log_alpha, dtype=torch.float32)
        advantages = q_values - values.detach()
        policy_loss = torch.mean(torch.exp(log_alpha_tensor) * log_probs - advantages)
        entropy_regularization = -torch.mean(torch.exp(log_alpha_tensor) * log_probs)
        sac_loss = value_loss + policy_loss + entropy_regularization
    
        sac_loss *= 0.025
        return sac_loss + loss
    
