import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from torch.cuda.amp import autocast
from ..GNNs.GCN import GCNconv


class NRNNCNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNCNAgent, self).__init__()
        self.args = args

        # GCN
        all_fcout = 0
        ene_fcout = 0
        gcnhid = 32
        gcnout = 32
        if self.args.use_graph:
            if self.args.toae == 'ae':
                all_fcout = 16
                ene_fcout = 12
                self.all_GCN = GCNconv(7, gcnhid, gcnout)
                self.all_GCN_fc = nn.Linear(self.args.n_agents * gcnout, all_fcout)
                self.ene_GCN = GCNconv(8, gcnhid, gcnout)
                self.ene_GCN_fc = nn.Linear(self.args.n_enemies * gcnout, ene_fcout)
            elif self.args.toae == 'a':
                all_fcout = 16
                self.all_GCN = GCNconv(7, gcnhid, gcnout)
                self.all_GCN_fc = nn.Linear(self.args.n_agents * gcnout, all_fcout)

        self.fc1 = nn.Linear(input_shape + all_fcout + ene_fcout,
                             args.rnn_hidden_dim) if self.args.use_graph else nn.Linear(input_shape,
                                                                                        args.rnn_hidden_dim)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, graph_input, hidden_state):
        b, a, e = inputs[0].size()
        num = e - inputs[1]

        inputs = inputs[0].view(-1, e)

        if self.args.use_graph:
            all_data, ene_data = graph_input
            all_W = F.sigmoid(self.all_GCN_fc(self.all_GCN(all_data).reshape(b * a, -1)))
            if self.args.toae == 'ae':
                ene_W = F.sigmoid(self.ene_GCN_fc(self.ene_GCN(ene_data).reshape(b * a, -1)))
                inputs = th.cat((inputs[:, :num], all_W, ene_W, inputs[:, num:]), dim=-1)
            else:
                inputs = th.cat((inputs[:, :num], all_W, inputs[:, num:]), dim=-1)



        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)