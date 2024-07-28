import torch
import torch.nn as nn
from torchdiffeq import odeint
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from thop import profile
import sys


import plotly.graph_objects as go
from plotly.subplots import make_subplots


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  
src_directory = os.path.join(current_directory, '../src')
data_directory = os.path.join(current_directory, 'data')
result_directory = os.path.join(current_directory, 'result')
sys.path.append(src_directory)
import networks
import model_utils



EVAL_MODE = True
# EVAL_MODE = False


# constraint function g(u)
def constraint_fn(u):
    bs, _ = u.shape
    n = u.shape[1] // 2
    u_reshaped = u.reshape(bs, 2, n)
    U, V = u_reshaped[:, 0, :], u_reshaped[:, 1, :]
    energy = 0.5 * torch.sum(9.8 * U**2 + V**2, dim=1)
    diff = torch.abs(0.1 - energy)
    return diff




def manifolf_constraint_fn(u):
    bs, _ = u.shape
    n = u.shape[1] // 2
    u_reshaped = u.reshape(bs, 2, n)
    U, V = u_reshaped[:, 0, :], u_reshaped[:, 1, :]
    energy = 0.5 * torch.sum(9.8 * U**2 + V**2, dim=1)
    
    target_energy_lower_bound = 0.1
    distance_to_manifold = torch.abs(target_energy_lower_bound - energy)
    
    distance_to_manifold = distance_to_manifold.unsqueeze(1).repeat(1, 2*n)
    return distance_to_manifold



def conservation_loss(predicted_seq):
    batch_size = predicted_seq.shape[0]
    time_step = predicted_seq.shape[1]
    predicted_uv_seq = predicted_seq.reshape(batch_size, time_step, 2, -1)
    u_seq = predicted_uv_seq[:, :, 0, :]
    v_seq = predicted_uv_seq[:, :, 1, :]
    
    conservation_value_seq = (9.8 * u_seq ** 2 + v_seq ** 2) * 0.5
    initial_conservation_value = conservation_value_seq[:, 0:1]
    conservation_value_diff = conservation_value_seq[:, 1:] - conservation_value_seq[:, :-1]
    # conservation_value_diff = conservation_value_seq - initial_conservation_value

    # conservation_loss = torch.mean(conservation_value_diff**2)

    threshold = 0.1
    diff = threshold - conservation_value_seq.sum(axis=2)
    loss = torch.abs(diff)
    conservation_loss = torch.mean(loss)
    return conservation_loss



def eval_model(model, model_name, vis_seq_index = 1, vis_ts=60):
    print()
    path = current_directory + '/' + model_name + '.pth'
    model = model_utils.load_model(model, path)
    model.eval()
    test_uv_data = np.load(data_directory + "/shallow_water_sequence_100_test.npy")
    test_uv_data = test_uv_data.reshape(batch_size, time_step, -1)
    initial_state = test_uv_data[:, 0, :]
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32).to(device) 
    x_tensor = initial_state_tensor
    predicted_states = model(x_tensor, t_span)

    model_utils.print_model_parameters(model, model_name)
    predicted_states_uv = predicted_states.reshape(batch_size, time_step, 2, -1)

    constrain_loss = conservation_loss(predicted_states_uv)
    predicted_states = predicted_states.cpu().detach().numpy().transpose(1, 0, 2)

    # print("Predicted states:")
    # print(predicted_states[0])
    # print("Ground truth states:")
    # print(test_uv_data[0])
    
    target_ts = time_step
    mse_loss = np.mean((predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :]) ** 2)
    l1_loss = np.mean(np.abs(predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :]))
    print(f"MSE Loss: {mse_loss}, L1 Loss: {l1_loss}, Model: " + model_name)
    print(f"Constrain Loss: {constrain_loss}")

    mse_losses = np.mean((predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :]) ** 2, axis=(0, 2))
    l1_losses = np.mean(np.abs(predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :]), axis=(0, 2))

    # plt.figure()
    # plt.plot(range(1, target_ts + 1), mse_losses)
    # plt.xlabel("Time Step")
    # plt.ylabel("MSE Loss")
    # plt.title(f"MSE Loss vs. Time Step ({model_name})")
    # plt.show()

    # plt.figure()
    # plt.plot(range(1, target_ts + 1), l1_losses)
    # plt.xlabel("Time Step")
    # plt.ylabel("L1 Loss")
    # plt.title(f"L1 Loss vs. Time Step ({model_name})")
    # plt.show()



    long_test_uv_data = np.load(data_directory + "/shallow_water_sequence_200.npy")
    long_time_step = long_test_uv_data.shape[1]
    long_test_uv_data = long_test_uv_data.reshape(batch_size, long_time_step, -1)
    initial_state = long_test_uv_data[:, 0, :]
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32).to(device) 
    long_t_span = torch.linspace(0, long_time_step/time_step, long_time_step)
    long_predicted_states = model(initial_state_tensor, long_t_span)
    long_constrain_loss = conservation_loss(long_predicted_states)
    long_predicted_states = long_predicted_states.cpu().detach().numpy().transpose(1, 0, 2)
    long_mse_loss = np.mean((long_predicted_states - long_test_uv_data) ** 2)
    long_l1_loss = np.mean(np.abs(long_predicted_states - long_test_uv_data))
    print(f"Longer Time Series - MSE Loss: {long_mse_loss}, L1 Loss: {long_l1_loss}, Model: " + model_name)
    print(f"Longer Time Series - Constrain Loss: {long_constrain_loss}")

    # mape_loss = np.mean(np.abs((test_uv_data[:, :target_ts, :] - predicted_states[:, :target_ts, :]) / test_uv_data[:, :target_ts, :])) * 100
    # print(f"MAPE Loss: {mape_loss}")

    # rmse_loss = np.sqrt(np.mean((predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :]) ** 2))
    # print(f"RMSE Loss: {rmse_loss}")

    # mae_naive = np.mean(np.abs(test_uv_data[:, 1:target_ts, :] - test_uv_data[:, :target_ts-1, :]))
    # mase_loss = np.mean(np.abs(predicted_states[:, :target_ts, :] - test_uv_data[:, :target_ts, :])) / mae_naive
    # print(f"MASE Loss: {mase_loss}")

    if vis_ts > time_step:
        vis_ts = time_step

    predicted_u_and_v = predicted_states[vis_seq_index].reshape(time_step, 2, -1)
    test_u_and_v = test_uv_data[vis_seq_index].reshape(time_step, 2, -1)
    predicted_u = predicted_u_and_v[:vis_ts, 0]
    predicted_v = predicted_u_and_v[:vis_ts, 1]
    test_u = test_u_and_v[:vis_ts, 0]
    test_v = test_u_and_v[:vis_ts, 1]

    fig, ax = plt.subplots()

    line_pred_u, = ax.plot([], [], 'r-', label='Predicted U')
    line_pred_v, = ax.plot([], [], 'b-', label='Predicted V')
    line_test_u, = ax.plot([], [], 'r--', label='Test U')
    line_test_v, = ax.plot([], [], 'b--', label='Test V')


    ax.set_xlabel('Vector Index')
    ax.set_ylabel('Value')
    ax.legend()



    def update(frame):
        line_pred_u.set_data(range(len(predicted_u[frame])), predicted_u[frame])
        line_pred_v.set_data(range(len(predicted_v[frame])), predicted_v[frame])
        line_test_u.set_data(range(len(test_u[frame])), test_u[frame])
        line_test_v.set_data(range(len(test_v[frame])), test_v[frame])
        ax.set_title(f'{model_name} prediction: U and V')
        ax.set_xlim(0, len(test_u[frame])-1)
        ax.set_ylim(min(np.min(predicted_u), np.min(predicted_v), np.min(test_u), np.min(test_v)),
                max(np.max(predicted_u), np.max(predicted_v), np.max(test_u), np.max(test_v)))
        return line_pred_u, line_pred_v, line_test_u, line_test_v

    ani = FuncAnimation(fig, update, frames=range(vis_ts), blit=True)
    # plt.show()




    if vis_ts > time_step:
        vis_ts = time_step

    start_ts = 0
    end_ts = int(vis_ts * 1.0)

    predicted_u_and_v = predicted_states[vis_seq_index].reshape(time_step, 2, -1)
    test_u_and_v = test_uv_data[vis_seq_index].reshape(time_step, 2, -1)
    predicted_u = predicted_u_and_v[start_ts:end_ts, 0]
    predicted_v = predicted_u_and_v[start_ts:end_ts, 1]
    test_u = test_u_and_v[start_ts:end_ts, 0]
    test_v = test_u_and_v[start_ts:end_ts, 1]

    vector_length = len(test_u[0])
    t = np.arange(start_ts, end_ts)
    vector_indices = np.arange(vector_length)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted", "Ground Truth"),
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    fig.add_trace(go.Surface(x=-vector_indices, y=-t, z=predicted_u, name='Predicted U'), row=1, col=1)
    fig.add_trace(go.Surface(x=-vector_indices, y=-t, z=test_u, name='Test U', showscale=False), row=1, col=2)

    z_min = min(predicted_u.min(), test_u.min())
    z_max = max(predicted_u.max(), test_u.max())
    z_range = z_max - z_min
    z_tickvals = [z_min + z_range*0.3, z_min + z_range*0.5, z_min + z_range*0.7, z_max]
    z_ticktext = [f"{val:.2f}" for val in z_tickvals]

    fig.update_layout(scene=dict(
        xaxis_title='Vector Index',
        yaxis_title='Time Step',
        zaxis_title='Value',
        zaxis=dict(
            tickvals=z_tickvals,
            ticktext=z_ticktext
        ),
        aspectratio=dict(x=1, y=1, z=0.5),
    ),
        scene2=dict(
        xaxis_title='Vector Index',
        yaxis_title='Time Step',
        zaxis_title='Value',
        zaxis=dict(
            tickvals=z_tickvals,
            ticktext=z_ticktext
        ),
        aspectratio=dict(x=1, y=1, z=0.5),
    ),
        title=dict(
            text=f'{model_name} prediction: U (3D Visualization)',
            font=dict(
                family="Times New Roman",
                size=15,
                color="black"
            )
        ),
        width=1300, height=600,
        font=dict(
            family="Times New Roman",
            size=15,
            color="black"
        )
    )

    colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'red']]
    fig.update_traces(colorscale=colorscale, showscale=True, row=1, col=1)
    fig.update_traces(colorscale=colorscale, showscale=False, row=1, col=2)

    fig.write_image(current_directory + f"/{model_name}_prediction_U_3D.png", scale=3)

    return mse_losses, l1_losses




def compute_constrain_level(u, eta):
    potential = 0.5 * 9.8 * eta.pow(2).sum(dim = 0)
    kinetic = 0.5 * 1.0 * u.pow(2).sum(dim = 0)
    total_energy = potential + kinetic
    return total_energy

def constrain_level(model, model_name):


    path = current_directory + '/' + model_name + '.pth'
    model = model_utils.load_model(model, path)
    model.eval()
    test_uv_data = np.load(data_directory + "/shallow_water_sequence_100_test.npy")
    test_uv_data = test_uv_data.reshape(batch_size, time_step, -1)
    initial_state = test_uv_data[:, 0, :]
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32).to(device) 
    x_tensor = initial_state_tensor
    predicted_states = model(x_tensor, t_span)
    
    predicted_states = predicted_states.cpu().detach().numpy().transpose(1, 0, 2)
    predicted_states = predicted_states.reshape(batch_size, time_step, 2, -1)

    energy_data = []
    for traj in predicted_states:
        traj_energy = []
        for t in range(time_step):
            u, v = traj[t]
            energy = compute_constrain_level(torch.from_numpy(u), torch.from_numpy(v))
            traj_energy.append(energy.item())
        energy_data.append(traj_energy)

    return np.array(energy_data)



def gt_constrain_level():
    test_uv_data = np.load(data_directory + "/shallow_water_sequence_100_test.npy")
    energy_data = []
    for traj in test_uv_data:
        traj_energy = []
        for t in range(time_step):
            u, v = traj[t]
            energy = compute_constrain_level(torch.from_numpy(u), torch.from_numpy(v))
            traj_energy.append(energy.item())
        energy_data.append(traj_energy)

    return np.array(energy_data)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uv_data = np.load(data_directory + "/shallow_water_sequence_100.npy")
    batch_size = uv_data.shape[0] 
    time_step = uv_data.shape[1]
    uv_data = uv_data.reshape(batch_size, time_step, -1)

    input_dim = uv_data.shape[2]
    print(input_dim)
    gamma = 1.0
    # num_epochs = 100
    # learning_rate = 3e-3
    num_epochs = 100
    learning_rate = 6e-3
    t_span = torch.linspace(0, 1, time_step)
    x0 = uv_data[:, 0, :]



    test_uv_data = np.load(data_directory + "/shallow_water_sequence_100_test.npy")
    test_uv_data = test_uv_data.reshape(batch_size, time_step, -1)
    test_x0 = test_uv_data[:, 0, :]




    trainer = model_utils.ModelTrainer(x0, t_span, uv_data, test_x0, test_uv_data, num_epochs, learning_rate, device)


    hidden_dim = 135
    num_layers = 2
    node = networks.NeuralODE(input_dim, hidden_dim, num_layers).to(device)
    node_path = current_directory + '/NODE.pth'

    

    hidden_dim = 32
    num_layers = 1
    mlp_hidden_dim = 128 
    mlp_num_layers = 2
    node_mlp = networks.NeuralODE_MLP(input_dim, hidden_dim, num_layers, mlp_hidden_dim, mlp_num_layers).to(device)
    node_mlp_path = current_directory + '/NODE_MLP.pth'



    hidden_dim = 135
    num_layers = 2
    output_dim = input_dim
    constraint_dim = 1
    f_model = networks.ODEFunc(input_dim, hidden_dim, num_layers).to(device)
    snde = networks.SNDE(f_model, constraint_fn, gamma).to(device)
    snde_path = current_directory + '/SNDE.pth'


    hidden_dim = 32
    num_layers = 1
    sac_hidden_dim = 64 
    sac_num_layers=2
    node_sac = networks.NeuralODE_SAC(input_dim, hidden_dim, num_layers, sac_hidden_dim, sac_num_layers).to(device)
    node_sac_path = current_directory + '/NODE_SAC.pth'
    
    hidden_dim = 32
    num_layers = 1
    sac_hidden_dim = 64 
    sac_num_layers=2
    node_sac_free = networks.NeuralODE_SAC(input_dim, hidden_dim, num_layers, sac_hidden_dim, sac_num_layers).to(device)
    node_sac_free_path = current_directory + '/NODE_SAC_NO_CONSTRAIN.pth'


    hidden_dim = 32
    num_layers = 1
    sac_hidden_dim = 64 
    sac_num_layers=2
    node_sac_m = networks.NeuralODE_SAC_Manifold(manifolf_constraint_fn, input_dim, hidden_dim, num_layers, sac_hidden_dim, sac_num_layers).to(device)
    node_sac_m_path = current_directory + '/NODE_SAC_MANIFOLD_CONSTRAIN.pth'


    hidden_dim = 32
    num_layers = 1
    sac_hidden_dim = 64 
    sac_num_layers=2
    node_sac_m_nl = networks.NeuralODE_SAC_Manifold(manifolf_constraint_fn, input_dim, hidden_dim, num_layers, sac_hidden_dim, sac_num_layers).to(device)
    node_sac_m_nl_path = current_directory + '/NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS.pth'



    hidden_dim = 32
    num_layers = 1
    sac_hidden_dim = 64
    sac_num_layers= 2
    node_sac_cont = networks.NeuralODE_SAC_CONTINUOUS(input_dim, hidden_dim, num_layers, sac_hidden_dim, sac_num_layers).to(device)
    node_sac_cont_path = current_directory + '/NODE_SAC_CONT.pth'




    losses_path = current_directory + "/losses"
    if not os.path.exists(losses_path):
        os.makedirs(losses_path)

    if not EVAL_MODE:
        node_sac_m_mse_l, node_sac_m_l1_l, node_sac_m_t_mse_l, node_sac_m_t_l1_l = trainer.train(node_sac_m, "NODE_SAC_MANIFOLD_CONSTRAIN", node_sac_m_path, conservation_loss)
        node_mse_l, node_l1_l, node_t_mse_l, node_t_l1_l = trainer.train(node, "NODE", node_path)
        node_mlp_mse_l, node_mlp_l1_l, node_mlp_t_mse_l, node_mlp_t_l1_l = trainer.train(node_mlp, "NODE_MLP", node_mlp_path)
        snde_mse_l, snde_l1_l, snde_t_mse_l, snde_t_l1_l = trainer.train(snde, "SNDE", snde_path)
        node_sac_free_mse_l, node_sac_free_l1_l, node_sac_free_t_mse_l, node_sac_free_t_l1_l = trainer.train(node_sac_free, "NODE_SAC_NO_CONSTRAIN", node_sac_free_path)
        node_sac_mse_l, node_sac_l1_l, node_sac_t_mse_l, node_sac_t_l1_l = trainer.train(node_sac, "NODE_SAC", node_sac_path, conservation_loss)
        node_sac_cont_mse_l, node_sac_cont_l1_l, node_sac_cont_t_mse_l, node_sac_cont_t_l1_l = trainer.train(node_sac_cont, "NODE_SAC_CONT", node_sac_cont_path)
        node_sac_m_nl_mse_l, node_sac_m_nl_l1_l, node_sac_m_nl_t_mse_l, node_sac_m_nl_t_l1_l = trainer.train(node_sac_m_nl, "NODE_SAC_MANIFOLD_CONSTRAIN_N0_LOSS", node_sac_m_path)


        np.save(os.path.join(losses_path, "node_sac_m_mse_l.npy"), node_sac_m_mse_l)
        np.save(os.path.join(losses_path, "node_sac_m_l1_l.npy"), node_sac_m_l1_l)
        np.save(os.path.join(losses_path, "node_sac_m_t_mse_l.npy"), node_sac_m_t_mse_l)
        np.save(os.path.join(losses_path, "node_sac_m_t_l1_l.npy"), node_sac_m_t_l1_l)


        np.save(os.path.join(losses_path, "node_sac_m_nl_mse_l.npy"), node_sac_m_nl_mse_l)
        np.save(os.path.join(losses_path, "node_sac_m_nl_l1_l.npy"), node_sac_m_nl_l1_l)
        np.save(os.path.join(losses_path, "node_sac_m_nl_t_mse_l.npy"), node_sac_m_nl_t_mse_l)
        np.save(os.path.join(losses_path, "node_sac_m_nl_t_l1_l.npy"), node_sac_m_nl_t_l1_l)
        
        np.save(os.path.join(losses_path, "node_mse_l.npy"), node_mse_l)
        np.save(os.path.join(losses_path, "node_l1_l.npy"), node_l1_l)
        np.save(os.path.join(losses_path, "node_t_mse_l.npy"), node_t_mse_l)
        np.save(os.path.join(losses_path, "node_t_l1_l.npy"), node_t_l1_l)
        
        np.save(os.path.join(losses_path, "node_mlp_mse_l.npy"), node_mlp_mse_l)
        np.save(os.path.join(losses_path, "node_mlp_l1_l.npy"), node_mlp_l1_l)
        np.save(os.path.join(losses_path, "node_mlp_t_mse_l.npy"), node_mlp_t_mse_l)
        np.save(os.path.join(losses_path, "node_mlp_t_l1_l.npy"), node_mlp_t_l1_l)
        
        np.save(os.path.join(losses_path, "snde_mse_l.npy"), snde_mse_l)
        np.save(os.path.join(losses_path, "snde_l1_l.npy"), snde_l1_l)
        np.save(os.path.join(losses_path, "snde_t_mse_l.npy"), snde_t_mse_l)
        np.save(os.path.join(losses_path, "snde_t_l1_l.npy"), snde_t_l1_l)
        
        np.save(os.path.join(losses_path, "node_sac_free_mse_l.npy"), node_sac_free_mse_l)
        np.save(os.path.join(losses_path, "node_sac_free_l1_l.npy"), node_sac_free_l1_l)
        np.save(os.path.join(losses_path, "node_sac_free_t_mse_l.npy"), node_sac_free_t_mse_l)
        np.save(os.path.join(losses_path, "node_sac_free_t_l1_l.npy"), node_sac_free_t_l1_l)
        
        np.save(os.path.join(losses_path, "node_sac_mse_l.npy"), node_sac_mse_l)
        np.save(os.path.join(losses_path, "node_sac_l1_l.npy"), node_sac_l1_l)
        np.save(os.path.join(losses_path, "node_sac_t_mse_l.npy"), node_sac_t_mse_l)
        np.save(os.path.join(losses_path, "node_sac_t_l1_l.npy"), node_sac_t_l1_l)

        
        np.save(os.path.join(losses_path, "node_sac_cont_mse_l.npy"), node_sac_cont_mse_l)
        np.save(os.path.join(losses_path, "node_sac_cont_l1_l.npy"), node_sac_cont_l1_l)
        np.save(os.path.join(losses_path, "node_sac_cont_t_mse_l.npy"), node_sac_cont_t_mse_l)
        np.save(os.path.join(losses_path, "node_sac_cont_t_l1_l.npy"), node_sac_cont_t_l1_l)
        


        

    node_sac_m_mse_l = np.load(os.path.join(losses_path, "node_sac_m_mse_l.npy"))
    node_sac_m_l1_l = np.load(os.path.join(losses_path, "node_sac_m_l1_l.npy"))
    node_sac_m_t_mse_l = np.load(os.path.join(losses_path, "node_sac_m_t_mse_l.npy"))
    node_sac_m_t_l1_l = np.load(os.path.join(losses_path, "node_sac_m_t_l1_l.npy"))


    node_sac_m_nl_mse_l = np.load(os.path.join(losses_path, "node_sac_m_nl_mse_l.npy"))
    node_sac_m_nl_l1_l = np.load(os.path.join(losses_path, "node_sac_m_nl_l1_l.npy"))
    node_sac_m_nl_t_mse_l = np.load(os.path.join(losses_path, "node_sac_m_nl_t_mse_l.npy"))
    node_sac_m_nl_t_l1_l = np.load(os.path.join(losses_path, "node_sac_m_nl_t_l1_l.npy"))

    node_mse_l = np.load(os.path.join(losses_path, "node_mse_l.npy"))
    node_l1_l = np.load(os.path.join(losses_path, "node_l1_l.npy"))
    node_t_mse_l = np.load(os.path.join(losses_path, "node_t_mse_l.npy"))
    node_t_l1_l = np.load(os.path.join(losses_path, "node_t_l1_l.npy"))

    node_mlp_mse_l = np.load(os.path.join(losses_path, "node_mlp_mse_l.npy"))
    node_mlp_l1_l = np.load(os.path.join(losses_path, "node_mlp_l1_l.npy"))
    node_mlp_t_mse_l = np.load(os.path.join(losses_path, "node_mlp_t_mse_l.npy"))
    node_mlp_t_l1_l = np.load(os.path.join(losses_path, "node_mlp_t_l1_l.npy"))

    snde_mse_l = np.load(os.path.join(losses_path, "snde_mse_l.npy"))
    snde_l1_l = np.load(os.path.join(losses_path, "snde_l1_l.npy"))
    snde_t_mse_l = np.load(os.path.join(losses_path, "snde_t_mse_l.npy"))
    snde_t_l1_l = np.load(os.path.join(losses_path, "snde_t_l1_l.npy"))

    node_sac_free_mse_l = np.load(os.path.join(losses_path, "node_sac_free_mse_l.npy"))
    node_sac_free_l1_l = np.load(os.path.join(losses_path, "node_sac_free_l1_l.npy"))
    node_sac_free_t_mse_l = np.load(os.path.join(losses_path, "node_sac_free_t_mse_l.npy"))
    node_sac_free_t_l1_l = np.load(os.path.join(losses_path, "node_sac_free_t_l1_l.npy"))

    node_sac_mse_l = np.load(os.path.join(losses_path, "node_sac_mse_l.npy"))
    node_sac_l1_l = np.load(os.path.join(losses_path, "node_sac_l1_l.npy"))
    node_sac_t_mse_l = np.load(os.path.join(losses_path, "node_sac_t_mse_l.npy"))
    node_sac_t_l1_l = np.load(os.path.join(losses_path, "node_sac_t_l1_l.npy"))

    node_sac_cont_mse_l = np.load(os.path.join(losses_path, "node_sac_cont_mse_l.npy"))
    node_sac_cont_l1_l = np.load(os.path.join(losses_path, "node_sac_cont_l1_l.npy"))
    node_sac_cont_t_mse_l = np.load(os.path.join(losses_path, "node_sac_cont_t_mse_l.npy"))
    node_sac_cont_t_l1_l = np.load(os.path.join(losses_path, "node_sac_cont_t_l1_l.npy"))


    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    axs[0, 0].plot(node_sac_m_mse_l, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    axs[0, 0].plot(node_sac_m_nl_mse_l, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")
    axs[0, 0].plot(node_mse_l, label="NODE")
    axs[0, 0].plot(node_mlp_mse_l, label="NODE_MLP")
    axs[0, 0].plot(snde_mse_l, label="SNDE")
    axs[0, 0].plot(node_sac_free_mse_l, label="NODE_SAC_NO_CONSTRAIN")
    axs[0, 0].plot(node_sac_mse_l, label="NODE_SAC")
    axs[0, 0].set_title("MSE Loss Comparison")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(node_sac_m_l1_l, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    axs[0, 1].plot(node_sac_m_nl_l1_l, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")
    axs[0, 1].plot(node_l1_l, label="NODE")
    axs[0, 1].plot(node_mlp_l1_l, label="NODE_MLP")
    axs[0, 1].plot(snde_l1_l, label="SNDE")
    axs[0, 1].plot(node_sac_free_l1_l, label="NODE_SAC_NO_CONSTRAIN")
    axs[0, 1].plot(node_sac_l1_l, label="NODE_SAC")
    axs[0, 1].set_title("L1 Loss Comparison")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("L1 Loss")
    axs[0, 1].legend()


    axs[1, 0].plot(node_sac_m_t_mse_l, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    axs[1, 0].plot(node_sac_m_nl_t_mse_l, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")
    axs[1, 0].plot(node_t_mse_l, label="NODE")
    axs[1, 0].plot(node_mlp_t_mse_l, label="NODE_MLP")
    axs[1, 0].plot(snde_t_mse_l, label="SNDE")
    axs[1, 0].plot(node_sac_free_t_mse_l, label="NODE_SAC_NO_CONSTRAIN")
    axs[1, 0].plot(node_sac_t_mse_l, label="NODE_SAC")
    axs[1, 0].set_title("Test MSE Loss Comparison")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Test MSE Loss")
    axs[1, 0].legend()

    axs[1, 1].plot(node_sac_m_t_l1_l, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    axs[1, 1].plot(node_sac_m_nl_t_l1_l, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")
    axs[1, 1].plot(node_t_l1_l, label="NODE")
    axs[1, 1].plot(node_mlp_t_l1_l, label="NODE_MLP")
    axs[1, 1].plot(snde_t_l1_l, label="SNDE")
    axs[1, 1].plot(node_sac_free_t_l1_l, label="NODE_SAC_NO_CONSTRAIN")
    axs[1, 1].plot(node_sac_t_l1_l, label="NODE_SAC")
    axs[1, 1].set_title("Test MSE Loss Comparison")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Test MSE Loss")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig = make_subplots(rows=2, cols=2, subplot_titles=("MSE Loss Comparison", "L1 Loss Comparison", 
                                                        "Test MSE Loss Comparison", "Test L1 Loss Comparison"),
                        horizontal_spacing=0.15, vertical_spacing=0.15)

    # MSE Loss Comparison
    fig.add_trace(go.Scatter(x=list(range(len(node_sac_m_mse_l))), y=node_sac_m_mse_l, name="NODE-SAC", line=dict(color=colors[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(node_mse_l))), y=node_mse_l, name="NODE", line=dict(color=colors[1])), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(node_mlp_mse_l))), y=node_mlp_mse_l, name="NODE-MLP", line=dict(color=colors[2])), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(snde_mse_l))), y=snde_mse_l, name="SNDE", line=dict(color=colors[3])), row=1, col=1)

    # L1 Loss Comparison
    fig.add_trace(go.Scatter(x=list(range(len(node_sac_m_l1_l))), y=node_sac_m_l1_l, name="NODE-SAC", line=dict(color=colors[0]), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(node_l1_l))), y=node_l1_l, name="NODE", line=dict(color=colors[1]), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(node_mlp_l1_l))), y=node_mlp_l1_l, name="NODE-MLP", line=dict(color=colors[2]), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(snde_l1_l))), y=snde_l1_l, name="SNDE", line=dict(color=colors[3]), showlegend=False), row=1, col=2)

    # Test MSE Loss Comparison
    fig.add_trace(go.Scatter(x=list(range(len(node_sac_m_t_mse_l))), y=node_sac_m_t_mse_l, name="NODE-SAC", line=dict(color=colors[0]), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(node_t_mse_l))), y=node_t_mse_l, name="NODE", line=dict(color=colors[1]), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(node_mlp_t_mse_l))), y=node_mlp_t_mse_l, name="NODE-MLP", line=dict(color=colors[2]), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(snde_t_mse_l))), y=snde_t_mse_l, name="SNDE", line=dict(color=colors[3]), showlegend=False), row=2, col=1)

    # Test L1 Loss Comparison
    fig.add_trace(go.Scatter(x=list(range(len(node_sac_m_t_l1_l))), y=node_sac_m_t_l1_l, name="NODE-SAC", line=dict(color=colors[0]), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(node_t_l1_l))), y=node_t_l1_l, name="NODE", line=dict(color=colors[1]), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(node_mlp_t_l1_l))), y=node_mlp_t_l1_l, name="NODE-MLP", line=dict(color=colors[2]), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(snde_t_l1_l))), y=snde_t_l1_l, name="SNDE", line=dict(color=colors[3]), showlegend=False), row=2, col=2)


    fig.update_layout(
        height=700,
        width=1050,
        title_text="Loss Comparison",
        showlegend=True,
        font=dict(
            family="Times New Roman",
            size=14,
            color="black"
        )
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1, title_standoff=3)
    fig.update_xaxes(title_text="Epoch", row=1, col=2, title_standoff=3)
    fig.update_xaxes(title_text="Epoch", row=2, col=1, title_standoff=3)
    fig.update_xaxes(title_text="Epoch", row=2, col=2, title_standoff=3)
    fig.update_yaxes(title_text="MSE Loss", row=1, col=1, title_standoff=3)
    fig.update_yaxes(title_text="L1 Loss", row=1, col=2, title_standoff=3)
    fig.update_yaxes(title_text="Test MSE Loss", row=2, col=1, title_standoff=3)
    fig.update_yaxes(title_text="Test L1 Loss", row=2, col=2, title_standoff=3)

    fig.show()
    fig.write_image(result_directory + "/loss_comparison.png", scale=2)



    title_standoff = 2.5
    height = 450
    width = 600

    marker_symbol = 'circle' 
    marker_size = 8

    def sparsify(arr, n=20):
        if len(arr) <= n:
            return arr
        else:
            step = len(arr) // n
            idxs = list(range(0, len(arr), step))
            return [arr[i] for i in idxs]

    # MSE Loss Comparison
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_sac_m_mse_l)))), y=sparsify(node_sac_m_mse_l), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_mse_l)))), y=sparsify(node_mse_l), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_mlp_mse_l)))), y=sparsify(node_mlp_mse_l), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse.add_trace(go.Scatter(x=sparsify(list(range(len(snde_mse_l)))), y=sparsify(snde_mse_l), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol))) 
    fig_mse.update_layout(title="MSE Loss Comparison", xaxis=dict(title="Epoch", title_standoff=title_standoff), yaxis=dict(title="MSE Loss", title_standoff=title_standoff),
                font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_mse.write_image(result_directory + "/mse_loss_comparison.png", scale=2)

    # L1 Loss Comparison 
    fig_l1 = go.Figure()
    fig_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_sac_m_l1_l)))), y=sparsify(node_sac_m_l1_l), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_l1_l)))), y=sparsify(node_l1_l), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_mlp_l1_l)))), y=sparsify(node_mlp_l1_l), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1.add_trace(go.Scatter(x=sparsify(list(range(len(snde_l1_l)))), y=sparsify(snde_l1_l), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1.update_layout(title="L1 Loss Comparison", xaxis=dict(title="Epoch", title_standoff=title_standoff), yaxis=dict(title="L1 Loss", title_standoff=title_standoff),
                            font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_l1.write_image(result_directory + "/l1_loss_comparison.png", scale=2)

    # Test MSE Loss Comparison
    fig_test_mse = go.Figure()
    fig_test_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_sac_m_t_mse_l)))), y=sparsify(node_sac_m_t_mse_l), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_t_mse_l)))), y=sparsify(node_t_mse_l), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_mse.add_trace(go.Scatter(x=sparsify(list(range(len(node_mlp_t_mse_l)))), y=sparsify(node_mlp_t_mse_l), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_mse.add_trace(go.Scatter(x=sparsify(list(range(len(snde_t_mse_l)))), y=sparsify(snde_t_mse_l), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_mse.update_layout(title="Test MSE Loss Comparison", xaxis=dict(title="Epoch", title_standoff=title_standoff), yaxis=dict(title="Test MSE Loss", title_standoff=title_standoff),
                                font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_test_mse.write_image(result_directory + "/test_mse_loss_comparison.png", scale=2)

    # Test L1 Loss Comparison
    fig_test_l1 = go.Figure()
    fig_test_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_sac_m_t_l1_l)))), y=sparsify(node_sac_m_t_l1_l), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_t_l1_l)))), y=sparsify(node_t_l1_l), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_l1.add_trace(go.Scatter(x=sparsify(list(range(len(node_mlp_t_l1_l)))), y=sparsify(node_mlp_t_l1_l), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_l1.add_trace(go.Scatter(x=sparsify(list(range(len(snde_t_l1_l)))), y=sparsify(snde_t_l1_l), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_test_l1.update_layout(title="Test L1 Loss Comparison", xaxis=dict(title="Epoch", title_standoff=title_standoff), yaxis=dict(title="Test L1 Loss", title_standoff=title_standoff),
                                font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_test_l1.write_image(result_directory + "/test_l1_loss_comparison.png", scale=2)








    gt_constrain_level = gt_constrain_level()

    node_constrain_level = constrain_level(node, "NODE")
    snde_constrain_level = constrain_level(snde, "SNDE")
    node_sac_constrain_level = constrain_level(node_sac_m, "NODE_SAC_MANIFOLD_CONSTRAIN")
    
    mean_gt_constrain_level = (np.max(gt_constrain_level, axis=1) + np.min(gt_constrain_level, axis=1)) / 2
    mean_node_constrain_level = (np.max(node_constrain_level, axis=1) + np.min(node_constrain_level, axis=1)) / 2
    mean_snde_constrain_level = (np.max(snde_constrain_level, axis=1) + np.min(snde_constrain_level, axis=1)) / 2
    mean_node_sac_constrain_level = (np.max(node_sac_constrain_level, axis=1) + np.min(node_sac_constrain_level, axis=1)) / 2
    
    # mean_gt_constrain_level = np.mean(gt_constrain_level, axis=1)
    # mean_snde_constrain_level = np.mean(snde_constrain_level, axis=1)
    # mean_node_sac_constrain_level = np.mean(node_sac_constrain_level, axis=1)
    # mean_node_sac_constrain_level = np.mean(node_sac_constrain_level, axis=1)





    # plt.figure(figsize=(8, 6))

    # plt.plot(range(batch_size), mean_gt_constrain_level, '-', color='black', linewidth=2, label='Energy Constrain')

    # plt.plot(range(batch_size), mean_node_constrain_level, 'o', color='red', markersize=5, label='NODE')
    # plt.plot(range(batch_size), mean_snde_constrain_level, 's', color='blue', markersize=5, label='SNDE')
    # plt.plot(range(batch_size), mean_node_sac_constrain_level, '^', color='green', markersize=5, label='NODE_SAC')

    # plt.xlabel('Trajectory Index')
    # plt.ylabel('Wave Energy Level')
    # plt.title('Comparison of Wave Energy Levels for Different Models')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(current_directory + "/constrain_level.png")
    # # plt.show()


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(batch_size)),
        y=mean_gt_constrain_level,
        mode='lines',
        name='Wave Energy Level Constrain',
        line=dict(color='red', width=7)
    ))

    fig.add_trace(go.Scatter(
        x=list(range(batch_size)),
        y=mean_node_constrain_level,
        mode='markers',
        name='NODE',
        marker=dict(color='orange', size=7, opacity=0.7)
    ))

    fig.add_trace(go.Scatter(
        x=list(range(batch_size)),
        y=mean_snde_constrain_level,
        mode='markers',
        name='SNDE',
        marker=dict(color='blue', size=7, opacity=0.7, symbol='square')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(batch_size)),
        y=mean_node_sac_constrain_level,
        mode='markers',
        name='NODE_SAC',
        marker=dict(color='green', size=7, opacity=0.7, symbol='triangle-up')
    ))

    fig.update_layout(
        title=dict(
            text='Wave Energy Levels for Different Methods in Shallow Water Modeling',
            font=dict(family='Times New Roman', size=21, color='black')
        ),
        xaxis=dict(
            title=dict(text='Trajectory Index', font=dict(family='Times New Roman', size=22, color='black')),
            tickfont=dict(family='Times New Roman', size=18, color='black')
        ),
        yaxis=dict(
            title=dict(text='Wave Energy Level', font=dict(family='Times New Roman', size=22, color='black')),
            tickfont=dict(family='Times New Roman', size=18, color='black')
        ),
        legend=dict(font=dict(family='Times New Roman', size=18, color='black')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )


    # fig.show()


    fig.write_image(current_directory + "/constrain_level.png", scale = 3)







    node_mse_losses, node_l1_losses = eval_model(node, "NODE", 0, time_step)
    node_mlp_mse_losses, node_mlp_l1_losses = eval_model(node_mlp, "NODE_MLP", 0, time_step)
    snde_mse_losses, snde_l1_losses = eval_model(snde, "SNDE", 0, time_step)
    node_sac_free_mse_losses, node_sac_free_l1_losses = eval_model(node_sac_free, "NODE_SAC_NO_CONSTRAIN", 0, time_step)
    node_sac_mse_losses, node_sac_l1_losses = eval_model(node_sac, "NODE_SAC", 0, time_step)

    node_sac_m_mse_losses, node_sac_m_l1_losses = eval_model(node_sac_m, "NODE_SAC_MANIFOLD_CONSTRAIN", 0, time_step)
    # node_sac_m_nl_mse_losses, node_sac_m_nl_l1_losses = eval_model(node_sac_m_nl, "NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS", 0, time_step)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, time_step + 1), node_mse_losses, label="NODE")
    plt.plot(range(1, time_step + 1), node_mlp_mse_losses, label="NODE_MLP")
    plt.plot(range(1, time_step + 1), snde_mse_losses, label="SNDE")
    plt.plot(range(1, time_step + 1), node_sac_free_mse_losses, label="NODE_SAC_NO_CONSTRAIN")
    plt.plot(range(1, time_step + 1), node_sac_mse_losses, label="NODE_SAC")
    plt.plot(range(1, time_step + 1), node_sac_m_mse_losses, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    # plt.plot(range(1, time_step + 1), node_sac_m_nl_mse_losses, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")

    plt.xlabel("Time Step")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss vs. Time Step")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, time_step + 1), node_l1_losses, label="NODE")
    plt.plot(range(1, time_step + 1), node_mlp_l1_losses, label="NODE_MLP")
    plt.plot(range(1, time_step + 1), snde_l1_losses, label="SNDE")
    plt.plot(range(1, time_step + 1), node_sac_free_l1_losses, label="NODE_SAC_NO_CONSTRAIN")
    plt.plot(range(1, time_step + 1), node_sac_l1_losses, label="NODE_SAC")
    plt.plot(range(1, time_step + 1), node_sac_m_l1_losses, label="NODE_SAC_MANIFOLD_CONSTRAIN")
    # plt.plot(range(1, time_step + 1), node_sac_m_nl_l1_losses, label="NODE_SAC_MANIFOLD_CONSTRAIN_NO_LOSS")
    plt.xlabel("Time Step")
    plt.ylabel("L1 Loss")
    plt.title("L1 Loss vs. Time Step")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_comparison.png")
    plt.show()

    # Time step comparison
    time_steps = list(range(1, time_step + 1))


    
    # MSE Loss vs. Time Step
    fig_mse_time = go.Figure()
    fig_mse_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_mse_losses), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_mlp_mse_losses), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(snde_mse_losses), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_sac_m_mse_losses), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_mse_time.update_layout(title="MSE Loss vs. Time Step", xaxis=dict(title="Time Step", title_standoff=title_standoff), yaxis=dict(title="MSE Loss", title_standoff=title_standoff),
                            font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_mse_time.write_image(result_directory + "/mse_loss_vs_time_step.png", scale=2)

    # L1 Loss vs. Time Step
    fig_l1_time = go.Figure()
    fig_l1_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_l1_losses), name="NODE", mode='lines+markers', line=dict(color=colors[1], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_mlp_l1_losses), name="NODE-MLP", mode='lines+markers', line=dict(color=colors[2], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(snde_l1_losses), name="SNDE", mode='lines+markers', line=dict(color=colors[3], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1_time.add_trace(go.Scatter(x=sparsify(time_steps), y=sparsify(node_sac_m_l1_losses), name="NODE-SAC", mode='lines+markers', line=dict(color=colors[0], width=3), marker=dict(size=marker_size, symbol=marker_symbol)))
    fig_l1_time.update_layout(title="L1 Loss vs. Time Step", xaxis=dict(title="Time Step", title_standoff=title_standoff), yaxis=dict(title="L1 Loss", title_standoff=title_standoff),
                            font=dict(family="Times New Roman", size=14, color="black"), height=height, width=width)
    fig_l1_time.write_image(result_directory + "/l1_loss_vs_time_step.png", scale=2)


    fig = make_subplots(rows=1, cols=2, subplot_titles=("MSE Loss vs. Time Step", "L1 Loss vs. Time Step"),
                        specs=[[{"secondary_y": True}, {"secondary_y": True}]])

    # MSE Loss vs. Time Step
    fig.add_trace(go.Scatter(x=time_steps, y=node_mse_losses, name="NODE", line=dict(color=colors[1])), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=node_mlp_mse_losses, name="NODE-MLP", line=dict(color=colors[2])), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=snde_mse_losses, name="SNDE", line=dict(color=colors[3])), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=node_sac_m_mse_losses, name="NODE-SAC", line=dict(color=colors[0])), row=1, col=1, secondary_y=False)

    # L1 Loss vs. Time Step
    fig.add_trace(go.Scatter(x=time_steps, y=node_l1_losses, name="NODE", line=dict(color=colors[1]), showlegend=False), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=node_mlp_l1_losses, name="NODE-MLP", line=dict(color=colors[2]), showlegend=False), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=snde_l1_losses, name="SNDE", line=dict(color=colors[3]), showlegend=False), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=time_steps, y=node_sac_m_l1_losses, name="NODE-SAC", line=dict(color=colors[0]), showlegend=False), row=1, col=2, secondary_y=False)

    fig.update_xaxes(title_text="Time Step", row=1, col=1, title_standoff=3)
    fig.update_xaxes(title_text="Time Step", row=1, col=2, title_standoff=3)
    fig.update_yaxes(title_text="MSE Loss", row=1, col=1, secondary_y=False, title_standoff=3)
    fig.update_yaxes(title_text="L1 Loss", row=1, col=2, secondary_y=False, title_standoff=3)

    fig.update_layout(title="Loss Comparison vs. Time Step", legend_title="Models",
                    font=dict(family="Times New Roman", size=14, color="black"), height=400, width=900, showlegend=True)

    fig.write_image(result_directory + "/loss_comparison_vs_time_step.png", scale=2) 
    fig.show()