import torch
import torch.nn as nn
import torch.nn.functional as F



class RRUCell(nn.Module):
    def __init__(self, num_units, output_size=256, relu_layers=1, middle_layer_size_multiplier=2, dropout_rate=0.5,
                 training=False, learable_parameter='false', S_init='constant'):
        super(RRUCell, self).__init__()

        self.num_units = num_units
        self.output_size = output_size
        self.relu_layers = relu_layers
        self.middle_layer_size_multiplier = middle_layer_size_multiplier
        self.dropout_rate = dropout_rate
        self.training = training
        self.learable_parameter = learable_parameter
        self.S_init = S_init
        self.residual_freq = 2

        self.bias_initializer = torch.zeros
        self.build()

    def build(self):
        input_depth = self.num_units
        total = input_depth + self.num_units
        n_middle_maps = round(self.middle_layer_size_multiplier * total)


        self.J_kernel = nn.ParameterList()
        self.J_bias = nn.ParameterList()
        if self.learable_parameter == 'lp':
            self.W_learable = nn.Parameter(torch.Tensor(n_middle_maps, total))
            nn.init.constant_(self.W_learable, 0.0)
            # nn.init.xavier_uniform_(self.W_learable)
            self.linear_learable = nn.Linear(total, n_middle_maps)

            for i in range(self.relu_layers):
                j_kernel = nn.Parameter(torch.Tensor(total, n_middle_maps))
                j_bias = nn.Parameter(torch.Tensor(n_middle_maps))

                nn.init.xavier_uniform_(j_kernel)
                nn.init.constant_(j_bias, 0)
                self.J_kernel.append(j_kernel)
                self.J_bias.append(j_bias)

        else:
            for i in range(self.relu_layers):
                if i == 0:
                    j_kernel = nn.Parameter(torch.Tensor(total, n_middle_maps))
                    j_bias = nn.Parameter(torch.Tensor(n_middle_maps))
                else:
                    j_kernel = nn.Parameter(torch.Tensor(n_middle_maps, n_middle_maps))
                    j_bias = nn.Parameter(torch.Tensor(n_middle_maps))
                nn.init.xavier_uniform_(j_kernel)
                nn.init.constant_(j_bias, 0)
                self.J_kernel.append(j_kernel)
                self.J_bias.append(j_bias)
        if self.S_init == 'constant':
            self.S_bias_variable = nn.Parameter(torch.Tensor(self.num_units))
            nn.init.constant_(self.S_bias_variable, 0.7)
        elif self.S_init == 'uniform_constant':
            self.mul_lr_multiplier = 10.
            weights = torch.empty(self.num_units)
            nn.init.uniform_(weights, 0.01, 0.99)
            weights.apply_(lambda x: (inv_sigmoid(torch.tensor(x))/self.mul_lr_multiplier).item())
            self.S_bias_variable = nn.Parameter(weights)
        elif self.S_init == 'uniform':
            self.S_bias_variable = nn.Parameter(torch.Tensor(self.num_units))
            # nn.init.xavier_uniform_(self.S_bias_variable)
            nn.init.uniform_(self.S_bias_variable)

        self.W_kernel = nn.Parameter(torch.Tensor(n_middle_maps, self.num_units + self.output_size))
        self.W_bias = nn.Parameter(torch.Tensor(self.num_units + self.output_size))
        nn.init.xavier_uniform_(self.W_kernel)
        nn.init.constant_(self.W_bias, 0)

        self.Z_ReZero = nn.Parameter(torch.Tensor(self.num_units))
        nn.init.constant_(self.Z_ReZero, 0)

        self.weight_relu0 = torch.tensor([0.25], requires_grad=True, device='cuda')
        self.weight_relu1 = torch.tensor([0.25], requires_grad=True, device='cuda')


    def forward(self, inputs, state):
        input_and_state = torch.cat([inputs, state], dim=1)

        j_start = input_and_state
        for i in range(self.relu_layers):
            after_j = torch.matmul(j_start, self.J_kernel[i]) + self.J_bias[i]
            if i == 0:
                #after_j = torch.layer_norm(after_j, )  TODO
                after_activation = F.prelu(instance_norm(after_j), self.weight_relu0)
            else:
                after_activation = F.prelu(after_j, self.weight_relu1) #F.relu(after_j)
            j_start = after_activation
        if self.training:
            dropout_rate = self.dropout_rate
        else:
            dropout_rate = 0.

        after_dropout = F.dropout(j_start, p=dropout_rate, training=self.training)

        after_w = torch.matmul(after_dropout, self.W_kernel) + self.W_bias

        output = after_w[:, self.num_units:]
        candidate = after_w[:, :self.num_units]

        if self.S_init == 'constant':
            final_state = state * torch.sigmoid(
                self.S_bias_variable) + candidate * self.Z_ReZero
        elif self.S_init == 'uniform_constant':
            final_state = state * torch.sigmoid(self.S_bias_variable * self.mul_lr_multiplier) + candidate * self.Z_ReZero
        elif self.S_init == 'uniform':
            final_state = state * torch.sigmoid(
                self.S_bias_variable) + candidate * self.Z_ReZero
        return output, final_state

    def zero_state(self, batch_size, dtype):
        initial = torch.cat([torch.sqrt(torch.tensor(self.num_units).float()) * 0.25 * torch.ones(1, dtype=dtype),
                             torch.zeros(self.num_units - 1, dtype=dtype)])
        value = torch.zeros(batch_size, self.num_units, dtype=dtype)
        return value + initial


def instance_norm(cur):
    variance = torch.mean(cur ** 2, dim=-1, keepdim=True)
    cur = cur * torch.rsqrt(variance + 1e-6)
    return cur

def inv_sigmoid(x):
    return torch.log(x / (1 - x))
