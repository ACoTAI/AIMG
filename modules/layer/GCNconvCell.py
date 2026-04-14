import torch.nn as nn
import torch
import math
from time import time


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # ²ÎÊýËæ»ú»¯
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input@self.weight
        # if not torch.all(adj == 0):
        # star = time()
        output = adj@support
        # print(time() - star)
        # star = time()
        # output = torch.cat([torch.spmm(adj[i, :, :], support[i, :, :]).unsqueeze(0) for i in range(input.shape[0])], dim=0)
        # print(time() - star)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MyGCN(nn.Module):
    def __init__(self,  input_dim: int, output_dim: int, ffd_drop=False,residual=True,**kwargs):
        super(MyGCN, self).__init__()

        #self._num_nodes
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.line = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.reset_parameters()
        self.ffd_drop = ffd_drop
        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim,output_dim, bias=True)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.line, gain=nn.init.calculate_gain("tanh"))

    def calculate_laplacian_with_self_loop(self, matrix):
        matrix = matrix + torch.eye(matrix.size(0)).to("cuda:0")
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

    def batch_calculate_laplacian_with_self_loop(self, matrix):
        batch_size = matrix.size(0)
        eye = torch.eye(matrix.size(1), device=matrix.device).unsqueeze(0)
        matrix = matrix + eye.expand(batch_size, -1, -1)

        row_sum = matrix.sum(2)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(1, 2).matmul(d_mat_inv_sqrt)
        return normalized_laplacian

    def forward(self, inputs, adj):

        # self.laplacian = torch.stack([self.calculate_laplacian_with_self_loop(adj[i, :, :]) for i in range(adj.shape[0])])

        self.laplacian = self.batch_calculate_laplacian_with_self_loop(adj)
        # self._num_nodes= adj.shape[0]
        #batch_size = inputs.shape[0]
        # (num_nodes, batch_size, feature)
        #inputs = inputs.transpose(0, 1)
        # (num_nodes, batch_size * feature)
        #inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
       # print("shapepp",self.laplacian.size())
        #print(inputs.size())
        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        #ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
       # ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        #outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        #print("size",outputs.size())
        if self.residual:
            outputs = outputs + self.lin_residual(inputs)
        # (batch_size, num_nodes, output_dim)
        #outputs = outputs.transpose(0, 1)
        return outputs, adj
