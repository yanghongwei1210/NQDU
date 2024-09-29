import torch
import torch.nn as nn
import numpy as np
class DataValueEstimator(nn.Module):
    def __init__(self, data_dim, label_dim, hidden_dim, layer_num, activation, comb_dim):
        super(DataValueEstimator, self).__init__()

        inter_layers = []
        inter_layers.append(nn.Linear(data_dim + label_dim, hidden_dim))
        inter_layers.append(getattr(nn, activation)())

        for i in range(layer_num - 3):
            inter_layers.append(nn.Linear(hidden_dim, hidden_dim))
            inter_layers.append(getattr(nn, activation)())

        inter_layers.append(nn.Linear(hidden_dim, comb_dim))
        inter_layers.append(getattr(nn, activation)())

        self.inter_layers = nn.Sequential(*inter_layers)

        comb_layers = []
        comb_layers.append(nn.Linear(comb_dim + label_dim, comb_dim))
        comb_layers.append(getattr(nn, activation)())
        comb_layers.append(nn.Linear(comb_dim, 1))
        comb_layers.append(nn.Sigmoid())

        self.comb_layers = nn.Sequential(*comb_layers)

    def forward(self, x_input, y_input, y_hat_input):
        inter_in = torch.cat((x_input, y_input), 1)
        inter_out = self.inter_layers(inter_in)
        comb_in = torch.cat((inter_out, y_hat_input), 1)
        comb_out = self.comb_layers(comb_in)
        return comb_out



class DVELoss(nn.Module):
    def __init__(self, epsilon, threshold):
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, est_data_value, s_input, reward_input):

        prob = torch.sum(s_input* torch.log(est_data_value + self.epsilon) + \
                             (1 - s_input) * \
                         torch.log(1 - est_data_value + self.epsilon))
        x = np.array(0)
        y = torch.from_numpy(x).float().to('cuda')
        dve_loss = (-reward_input * prob) + \
                   1e3 * (torch.maximum(torch.mean(est_data_value) - self.threshold, y) +
                          torch.maximum((1 - self.threshold) - torch.mean(est_data_value), y))

        return dve_loss



