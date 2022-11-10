"""
The based unit of graph convolutional networks., based on awesome previous work by https://github.com/yysijie/st-gcn
"""

import torch
import torch.nn as nn
from models.STG_NF.modules_pose import InvertibleConv1x1


class InvConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 LU_decomposed,
                 kernel_size):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)

    def forward(self, x, A, logdetA, logdet=None, reverse=False):
        if not reverse:
            x, dlogdet = self.conv(x, logdetA, reverse)
            n, kc, t, v = x.size()
            x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
            x_test = x.permute(0, 2, 3, 1, 4).reshape(n, kc // self.kernel_size, t, -1)
            a_test = A.reshape(-1, v)
            x = torch.cat((x[:, 0], x_test @ a_test), dim=1).reshape(x.shape)
            if logdet is not None:
                logdet = logdet + logdetA[1]
                return x, logdet
            return x, dlogdet + logdetA[1]
        else:
            a1_inverse = A[1].inverse()
            x0 = x[:, 0]
            x1 = (x[:, 1] - (x0 @ A[0])) @ a1_inverse
            x_rev = torch.cat((x0, x1), dim=1).reshape(x.shape)
            z, dlogdet = self.conv2d(x_rev, logdetA, reverse)
            if logdet is not None:
                logdet = logdet - logdetA[1]
                return z, logdet
            return z, dlogdet - logdetA[1]
