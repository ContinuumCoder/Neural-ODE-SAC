import torch
import torch.nn as nn
import numpy as np

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model, model_name):
    total_params = count_parameters(model)
    print(model_name + f" - total number of trainable parameters: {total_params}")


# def train_model(model, model_name, path, x0, t_span, gt, num_epochs, lr, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     t_span = t_span.to(device)
#     x0_tensor = torch.tensor(x0, dtype=torch.float32).to(device)
#     target = torch.tensor(gt, dtype=torch.float32).to(device)

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         u_pred = model(x0_tensor, t_span).permute(1, 0, 2)
#         loss = model.criterion(u_pred, target)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 50 == 0:
#             print(model_name + f" - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
#     save_model(model, path)



class ModelTrainer:
    def __init__(self, x0, t_span, gt, test_x0, test_gt, num_epochs, lr, device):
        self.x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        self.t_span = t_span.to(device)
        self.target = torch.tensor(gt, dtype=torch.float32).to(device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device

        self.test_x0 = torch.tensor(test_x0, dtype=torch.float32).to(device)
        self.test_target = torch.tensor(test_gt, dtype=torch.float32).to(device)
        self.mse_loss_fn = nn.MSELoss()
        self.l1_loss_fn = nn.L1Loss()

    def train(self, model, model_name, path, constrain_loss_func=None):
        print()
        mse_losses = []
        l1_losses = []
        test_mse_losses = []
        test_l1_losses = []
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            u_pred = model(self.x0, self.t_span).permute(1, 0, 2)
            loss = 0.0
            # if constrain_loss_func is None or epoch < self.num_epochs // 2:
            if constrain_loss_func is None:
                loss = model.criterion(u_pred, self.target)
            else:
                loss = model.criterion(u_pred, self.target, constrain_loss_func)
            mse_loss = self.mse_loss_fn(u_pred, self.target)
            l1_loss = self.l1_loss_fn(u_pred, self.target)
            loss.backward()
            mse_losses.append(mse_loss.cpu().detach().numpy())
            l1_losses.append(l1_loss.cpu().detach().numpy())
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"{model_name} - Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.7f}")
                model.eval()
                test_u_pred = model(self.test_x0, self.t_span).permute(1, 0, 2)
                test_mse_loss =  self.mse_loss_fn(test_u_pred, self.test_target)
                test_l1_loss =  self.l1_loss_fn(test_u_pred, self.test_target)
                print(f"Test: Mse Loss: {test_mse_loss.item():.7f}, L1 Loss: {test_l1_loss.item():.7f}")
                test_mse_losses.append(test_mse_loss.cpu().detach().numpy())
                test_l1_losses.append(test_l1_loss.cpu().detach().numpy())

        save_model(model, path)
        return np.array(mse_losses), np.array(l1_losses), np.array(test_mse_losses), np.array(test_l1_losses)