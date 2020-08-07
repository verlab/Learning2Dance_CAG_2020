import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
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
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
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
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.lrelu(x)
        # return self.relu(x), A

class Weight(nn.Module):
    def __init__(self,channels,output_nodes):
        super(Weight,self).__init__()
        # self.weight = torch.nn.Parameter(torch.rand(channels, output_nodes, requires_grad=True))
        self.weight = torch.nn.Parameter(torch.rand(2, output_nodes, requires_grad=True))

        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        # return torch.einsum('ncti,ki->ncti',(x,self.weight))
        return torch.einsum('kij,ki->kij',(x,self.weight))


class WeightD(nn.Module):
    def __init__(self,channels,output_nodes):
        super(WeightD,self).__init__()
        # self.weight = torch.nn.Parameter(torch.rand(channels, output_nodes, requires_grad=True))
        self.weight = torch.nn.Parameter(torch.rand(2, output_nodes, requires_grad=True))

        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        # return torch.einsum('ncti,ki->ncti',(x,self.weight))
        return torch.einsum('kji,ki->kij',(x,self.weight))

class UpSampling(nn.Module):

    def __init__(self,input_nodes,output_nodes,A,channels):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.A = A
        self.w = Weight(channels,output_nodes)

    def forward(self,x):
        assert x.size(3) == self.input_nodes
        assert self.A.size(0) == 2
        assert self.A.size(1) == self.output_nodes

        # res = torch.einsum('kij,njc->nic',(self.A,x))


        # res = torch.einsum('kij,nctj->ncti',(self.A,x))
        # res = self.w(res)
        
        res = self.w(self.A)
        res = torch.einsum('kij,nctj->ncti',(res,x))
        return res

class DownSampling(nn.Module):
    #need review
    def __init__(self,input_nodes,output_nodes,A,channels):
        super().__init__()
        self.A = A
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.w = WeightD(channels,output_nodes)

    def forward(self,x):
        assert x.size(3) == self.input_nodes
        assert self.A.size(0) == 2
        assert self.A.size(2) == self.output_nodes

        # res = torch.einsum('kij,ncti->nctj',(self.A,x))
        # res = self.w(res)
        # return res

        res = self.w(self.A)
        res = torch.einsum('kij,nctj->ncti',(res,x))
        return res
