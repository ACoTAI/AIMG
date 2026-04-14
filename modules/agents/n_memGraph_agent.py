import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from ..layer.RRUCell import RRUCell
from ..layer.self_atten import SelfAttention
from torch.cuda.amp import autocast
from itertools import permutations


class NGraphRRNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NGraphRRNAgent, self).__init__()
        self.args = args

        self.mem_number = args.mem_number
        self.mem_hidden_dim = args.rnn_hidden_dim * self.mem_number
        self.xhattenplus = args.xhattenplus
        if self.args.use_attenmem:
            self.emb_dim = 256
            if self.args.x_h_atten == 'xhatten':
                self.query = th.nn.Linear(self.args.rnn_hidden_dim, self.emb_dim)
                self.key = th.nn.Linear(self.args.rnn_hidden_dim, self.emb_dim)
                if self.xhattenplus:
                    self.value = th.nn.Linear(self.args.rnn_hidden_dim, self.emb_dim)
                    self.h_memmlp = nn.Linear(self.emb_dim*self.mem_number,  args.rnn_hidden_dim)
                    self.x_memmlp = nn.Linear(self.emb_dim, args.rnn_hidden_dim)

                self.softmax = th.nn.Softmax(dim=-2)
            elif self.args.x_h_atten == 'linear':
                self.memmlp = nn.Sequential(
                                nn.Linear(self.mem_hidden_dim, self.emb_dim),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Linear(self.emb_dim, self.emb_dim),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Linear(self.emb_dim, args.rnn_hidden_dim)
                            )

            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = RRUCell(args.rnn_hidden_dim, output_size=args.output_size, relu_layers=args.relu_layers,
                               training=args.training, learable_parameter=args.lp, S_init=args.S_init)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        else:
            # wm


            self.memfc = nn.Linear(self.mem_hidden_dim, args.rnn_hidden_dim)

            self.fc1 = nn.Linear(input_shape, self.mem_hidden_dim)
            self.rnn = RRUCell(self.mem_hidden_dim, output_size=args.output_size,  relu_layers=args.relu_layers, training=args.training, learable_parameter=args.lp,  S_init=args.S_init)
            self.fc2 = nn.Linear(self.mem_hidden_dim, args.n_actions)


        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def atten_mem(self, hidden, x):
        output = None
        if self.args.x_h_atten == 'xhatten':
            hidden = th.stack(hidden).transpose(0,2).transpose(0,1)
            hidden = hidden.reshape(-1,self.mem_number, self.args.rnn_hidden_dim)
            x = x.unsqueeze(-2)
            x = torch.cat((x, hidden), dim=-2)
            query = self.query(x)
            key = self.key(x)
            query = query / (x.shape[-1] ** (1 / 4))
            key = key / (x.shape[-1] ** (1 / 4))
            if self.xhattenplus:
                value = self.value(x)
                output = self.softmax(query@key.transpose(-1, -2))@value
                hidden = self.h_memmlp(output[:, 1:, :].reshape(-1, self.emb_dim*self.mem_number))

                output = torch.cat((self.x_memmlp(output[:,0,:]).unsqueeze(-2), hidden.unsqueeze(-2)), dim=-2)
            else:
                attention_weights = th.softmax(self.softmax(query@key.transpose(-1, -2))[:, 0, :][:, 1:], dim=-1).unsqueeze(-1)
                output = th.sum(hidden * attention_weights, dim=-2, keepdim=True).squeeze()
        elif self.args.x_h_atten == 'linear':
            output = self.memmlp(torch.cat(hidden, dim=-1).reshape(-1, self.mem_hidden_dim))
        return output

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def adjacency(self, inputs):
        b, _ = inputs.size()
        ally_node_feats = inputs[:,4:32].reshape(b, self.args.n_agents-1, -1)  #40, 4, 5
        enemy_node_feats = inputs[:,33:].reshape(b, self.args.n_enemies, -1)  #40, 6 ,5

        ally_adj = torch.zeros(b, self.args.n_agents-1, self.args.n_agents-1)
        enemy_adj = torch.zeros(b, self.args.args.n_enemies, self.args.args.n_enemies)

        for i in range(b):
            sigh_agent_index = []
            #ally
            for j in range(self.args.n_agents-1):
                if ally_node_feats[i, j, 0] != 0:
                    sigh_agent_index.append(j)
            if len(sigh_agent_index)>1:
                perms = np.array(permutations(sigh_agent_index, 2))
                ally_adj[b][perms] = 1
            #enemy
            for j in range(self.args.enemies):
                if ally_node_feats[i, j, 0] != 0:
                    sigh_agent_index.append(j)
            if len(sigh_agent_index)>1:
                perms = np.array(permutations(sigh_agent_index, 2))
                ally_adj[b][perms] = 1
        pass


    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)

        x = F.relu(self.fc1(inputs), inplace=True)


        if self.args.use_attenmem:
            # if getattr(self.args, "use_attenmem", False):
            h_in = self.atten_mem(hidden_state, x)
            if self.xhattenplus:
                x = h_in[:, 0, :]
                h_in = h_in[:, 1, :]
            # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        else:
            hidden_state = th.cat(hidden_state, dim=-1)
            h_in = hidden_state.reshape(-1, self.mem_hidden_dim)
        _, hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
            if not getattr(self.args, "use_attenmem", False):
                hh = self.memfc(self.layer_norm(hh))
        else:
            q = self.fc2(hh)
            if not getattr(self.args, "use_attenmem", False):
                hh = self.memfc(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)