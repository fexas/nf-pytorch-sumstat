# 他这个处理的好像不是tensor的说，只要处理vector就好
# 可以直接继承Flow类，context作为外部的输入
# 还有一个问题他是怎么优化的，应为他的参数都是nn.parameter，而我是写成了LearnableFunction的形式？

# 6.17
# 1. 用MLP代替LearnableFunciton，用MLP实例化w，b,u
# 2.  查看他的训练方式，使得其可以训练Flow

# Todo:
# 1. 查看示例中带有MLP的是如何初始化的
# 2.优化代码结构，传进参数的数量

# 感觉从理解上不太对，相当于是 我传进来batchsize就是sample_size了，然后context的维度是(context_size,d_context)


import numpy as np
import torch
from torch.nn import functional as F

from .base import Flow

from normflows.nets import MLP


class SumStatPlanar(Flow):
    """
    1. self应该包含的参数:
    p(压缩后的维度),MLP的初始化参数(layers,leaky,output_fn) # 我这个任务有必要设置output_fn吗？

    2. 需要外部传入的参数：
    d (z的维度), n*d_x 中的n和d_x(x/context的维度,context是matrix)
    """

    """
    w -  self.d*self.k ;
    z -  self.d*1 ;
    b -  self.k*1 ;
    u -  self.d*self.k

    x \in  R^{n \times d_x}
    z \in R^d
    T : R^{n \times d_x} \rightarrow R^p
    u: R^p \rightarrow R^{d \times k}
    w: R^p \rightarrow R^{d \times k}
    b: R^p \rightarrow R^k

    f(z) = z + u(T(x_{1:n})) * h(w(T(x_{1:n})) * z + b(T(x_{1:n}))) # h 代表tanh

    """

    def __init__(
        self,
        d_compress,  # 对应p
        d_context,  # 对应d_context
        d_flow_var,  # 对应 d (也是参数的dimension)
        sample_size,  # 对应n
        hidden_units,
        hidden_layers,
        leaky_T,
        leaky_w,
        leaky_u,
        leaky_b,
        output_fn_T=None,
        output_fn_w=None,
        output_fn_u=None,
        output_fn_b=None,
        act="tanh",
    ):
        super().__init__()
        self.d_compress = d_compress
        self.d_context = d_context
        self.sample_size = sample_size
        self.d_flow_var = d_flow_var
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.leaky_T = leaky_T
        self.leaky_w = leaky_w
        self.leaky_u = leaky_u
        self.leaky_b = leaky_b
        self.output_fn_T = output_fn_T
        self.output_fn_w = output_fn_w
        self.output_fn_u = output_fn_u
        self.output_fn_b = output_fn_b

        # 初始化需学习的参数（实际为神经网络）
        # 须知道的参数:
        # 1. n (x_{1:n})
        # 2. d_context
        # 3. p - compress_dim (dimension of T(x_{1:n}))
        # 4. d - dimesion of flow var z
        # 5. k
        self.T = MLP(
            [self.sample_size * self.d_context]
            + [self.hidden_units] * (hidden_layers - 1)
            + [self.d_compress],
            leaky_T,
            output_fn_T,
        )  #  T : R^{n \times d_context} \rightarrow R^p
        self.w = MLP(
            [self.d_compress]
            + [self.hidden_units] * (hidden_layers - 1)
            + [self.d_flow_var],
            leaky_w,
            output_fn_w,
        )  #  w : R^p \rightarrow R^{d}
        self.u = MLP(
            [self.d_compress]
            + [self.hidden_units] * (hidden_layers - 1)
            + [self.d_flow_var],
            leaky_u,
            output_fn_u,
        )  #  u : R^p \rightarrow R^{d}
        self.b = MLP(
            [self.d_compress] + [self.hidden_units] * (hidden_layers - 1) + [1],
            leaky_b,
            output_fn_b,
        )  #  b : R^p \rightarrow R^1

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("Nonlinearity is not implemented.")

    # context的维度是tensor,维度是（batch_size, n, d_context)
    # 例子：
    # context = torch.cat([torch.randn((batch_size, 2), device=device),
    #                      0.5 + 0.5 * torch.rand((batch_size, 2), device=device)],
    #                     dim=-1)

    def forward(self, z, context):

        context_reshaped = context.reshape(
            -1, self.sample_size * self.d_context
        )  # 将context的形状由 (-1, n, d_context) 变为 (-1, n * d_context,) （一维向量）
        T_context = self.T(context_reshaped)
        T_context = T_context.view(-1, self.d_compress)

        # 确保 T_context 是一个张量
        if not isinstance(T_context, torch.Tensor):
            raise TypeError("self.T 的输出不是一个 PyTorch 张量")

        w_T_context = self.w(T_context)
        b_T_context = self.b(T_context)
        u_T_context = self.u(T_context)

        # debug
        # print(
        #     "w_T_context,b_T_context,u_T_context:",
        #     w_T_context.shape,
        #     b_T_context.shape,
        #     u_T_context.shape,
        # )

        lin = (
            torch.sum(w_T_context * z, list(range(1, w_T_context.dim()))).unsqueeze(-1)
            + b_T_context
        )

        inner = torch.sum(
            w_T_context * u_T_context, list(range(1, w_T_context.dim()))
        ).unsqueeze(-1)
        # inner = torch.sum(w_T_context * u_T_context)

        u_parameter = torch.log(1 + torch.exp(inner)) - 1 - inner  # size(batch_size,1)
        u_parameter_expanded = u_parameter.expand(w_T_context.shape)
        w_divisor = torch.sum(
            w_T_context**2, list(range(1, w_T_context.dim()))
        ).unsqueeze(
            -1
        )  # shape should be (batch_size,1)
        w_divisor_expanded = w_divisor.expand(w_T_context.shape)
        u = (
            u_T_context + u_parameter_expanded * w_T_context / w_divisor_expanded
        )  # u.shape:(batch_size,d_flow_var) -- (500,4)

        # u = u_T_context + (
        #     torch.log(1 + torch.exp(inner)) - 1 - inner
        # ) * w_T_context / torch.sum(
        #     w_T_context**2
        # )  # constraint w.T * u > -1
        # h_ stands for h'
        if self.act == "tanh":
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        h_lin = self.h(lin)
        h_lin_expanded = h_lin.expand(u_T_context.shape)
        z_ = z + u_T_context * h_lin_expanded
        # z_ = z + u_T_context * self.h(lin)

        wT_u = torch.sum(w_T_context * u, list(range(1, w_T_context.dim()))).unsqueeze(
            -1
        )  # shape should be (batch_size,1) -- [500,1]

        log_det = torch.log(torch.abs(1 + wT_u * h_(lin))).squeeze()
        # log_det = torch.log(
        #     torch.abs(1 + torch.sum(w_T_context * u) * h_(lin.reshape(-1)))
        # )
        return z_, log_det

    def inverse(self, z, context):
        # Shape of variable:
        # z.shape = (batch_size,d_flow_var) -- (500,4)
        # context.shape = (batch_size,sample_size,d_context) -- (500,128,2)
        # w_T_context,b_T_context,u_T_context,z: ([500, 4]) ([500, 1]) ([500, 4]) ([500, 4])
        # lin : [500,1]
        # inner (w^t u): [500,1]
        if self.act != "leaky_relu":
            raise NotImplementedError("This flow has no algebraic inverse.")

        context_reshaped = context.reshape(
            -1, self.sample_size * self.d_context
        )  # 将context的形状由 (-1, n, d_context) 变为 (-1, n * d_context,) （一维向量）
        T_context = self.T(context_reshaped)
        T_context = T_context.view(-1, self.d_compress)

        # 确保 T_context 是一个张量
        if not isinstance(T_context, torch.Tensor):
            raise TypeError("self.T 的输出不是一个 PyTorch 张量")

        w_T_context = self.w(T_context)
        b_T_context = self.b(T_context)
        u_T_context = self.u(T_context)

        # debug
        # w_T_context,b_T_context,u_T_context,z: torch.Size([1, 4]) torch.Size([1, 1]) torch.Size([1, 4]) torch.Size([128, 4])
        # print(
        #     "w_T_context,b_T_context,u_T_context,z:",
        #     w_T_context.shape,
        #     b_T_context.shape,
        #     u_T_context.shape,
        #     z.shape,
        # )

        lin = (
            torch.sum(w_T_context * z, list(range(1, w_T_context.dim()))).unsqueeze(-1)
            + b_T_context
        )
        # print("linshape:", lin.shape)  # linshape: torch.Size([500,1])
        a = (lin < 0) * (
            self.h.negative_slope - 1.0
        ) + 1.0  # absorb leakyReLU slope into u
        # inner = torch.sum(w_T_context * u_T_context)
        inner = torch.sum(
            w_T_context * u_T_context, list(range(1, w_T_context.dim()))
        ).unsqueeze(-1)

        u_parameter = torch.log(1 + torch.exp(inner)) - 1 - inner  # size(batch_size,1)
        u_parameter_expanded = u_parameter.expand(w_T_context.shape)
        w_divisor = torch.sum(
            w_T_context**2, list(range(1, w_T_context.dim()))
        ).unsqueeze(
            -1
        )  # shape should be (batch_size,1)
        w_divisor_expanded = w_divisor.expand(w_T_context.shape)
        u = (
            u_T_context + u_parameter_expanded * w_T_context / w_divisor_expanded
        )  # u.shape:(batch_size,d_flow_var) -- (500,4)
        # u = u_T_context + (
        #     torch.log(1 + torch.exp(inner)) - 1 - inner
        # ) * w_T_context / torch.sum(w_T_context**2)
        # 此时(torch.log(1 + torch.exp(inner)) - 1 - inner)是个[500,1]的向量，没法直接乘形状为[500,4]的w_T_context,下面的除法应该也要是[500,4]
        dims = [-1] + (u.dim() - 1) * [1]
        # print(
        #     "inner,dims,ushape,ashape:", inner.shape, dims, u.shape, a.shape
        # )  # 原来的a.shape是torch.Size([1, 4])，
        # 确实 a的shape和linshape是一样的，为[500,1],没法和u[500,4]直接相乘，这里想做的应该是数乘，把a的1乘到四个分量上
        # 问题来了，哪来的500
        # 6.29留 -- 就是这个问题
        # u = a.reshape(*dims) * u  # debug
        # 修改这里，将a的形状调整为与u相同，然后逐元素相乘
        a_expanded = a.expand(u.shape)
        u = a_expanded * u  # debug
        inner_ = torch.sum(
            w_T_context * u, list(range(1, w_T_context.dim()))
        ).unsqueeze(
            -1
        )  # inner_shape:(batch_size,1) -- (500,1)
        # print("inner_shape:", inner.shape)
        # print("u.shape:", u.shape)
        # lin / (1 + inner_) should have shape:(batch_size,1)--(500,1)

        lin_parameter = lin / (1 + inner_)  # (batch_size,1)
        lin_parameter_expanded = lin_parameter.expand(u.shape)
        z_ = z - u * lin_parameter_expanded
        # z_ = z - u * (lin / (1 + inner_)).reshape(*dims)
        log_det = -torch.log(torch.abs(1 + inner_)).squeeze()  # shape:(batch_size,1)
        # print(
        #     "shape of lin_parameter,lin_parameter_expanded,u,z_,log_det:",
        #     lin_parameter.shape,
        #     lin_parameter_expanded.shape,
        #     u.shape,
        #     z_.shape,
        #     log_det.shape,
        # )
        return z_, log_det

    # def inverse(self, z, context):
    #     # debug
    #     print("z.shape", z.shape)
    #     print("context.shape", context.shape)

    #     if self.act != "leaky_relu":
    #         raise NotImplementedError("This flow has no algebraic inverse.")
    #     T_context = self.T(context.view(-1, self.sample_size * self.d_context)).view(
    #         -1, self.d_compress
    #     )

    #     # modify version

    #     w_T_context = self.w(T_context).view(-1, self.d_flow_var, self.d_k)

    #     w_T_context_transposed = torch.transpose(w_T_context, 1, 2)

    #     # debug
    #     print("w_T_context_transposed :", w_T_context_transposed.shape)

    #     # 扩展 z 的维度以匹配 w_T_x_transposed 的中间维度（即 self.d）
    #     z_expanded = z.unsqueeze(2)  # 扩展维度后形状为 (-1, self.d，1)
    #     # debug
    #     print("z_expanded:", z_expanded.shape)

    #     lin = torch.matmul(w_T_context_transposed, z_expanded)
    #     lin = lin.squeeze(-1)  # 如果需要，可以移除最后一个维度, 结果形状为 (-1, self.k)
    #     lin = lin + self.b(T_context)  # 形状为(-1,self.k)
    #     # debug
    #     print("lin:", lin.shape)

    #     a = (lin < 0) * (
    #         self.h.negative_slope - 1.0
    #     ) + 1.0  # absorb leakyReLU slope into u

    #     # debug
    #     u_T_context = self.u(T_context)
    #     print("u_T_context.shape:", u_T_context.shape)

    #     inner = torch.sum(self.w(T_context) * self.u(T_context))
    #     u = self.u(T_context) + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w(
    #         T_context
    #     ) / torch.sum(self.w(T_context) ** 2)
    #     dims = [-1] + (u.dim() - 1) * [1]
    #     u = a.reshape(*dims) * u
    #     inner_ = torch.sum(self.w(T_context) * u, dim=1)

    #     # modify version
    #     u = u.view(-1, self.d_flow_var, self.d_k)
    #     # 出问题的地方
    #     z_ = z - torch.matmul(u, lin.unsqueeze(2)) / (
    #         1 + inner_
    #     )  # 将 lin的形状变为(-1,self.k，1)
    #     log_det = -torch.log(torch.abs(1 + inner_))
    #     return z_, log_det

    # def forward(self, z, context):

    #     context_reshaped = context.reshape(
    #         -1, self.sample_size * self.d_context
    #     )  # 将context的形状由 (-1, n, d_context) 变为 (-1, n * d_context,) （一维向量）
    #     T_context = self.T(context_reshaped)
    #     T_context = T_context.view(-1, self.d_compress)

    #     w_T_context = self.w(T_context)

    #     # 确保 T_context 是一个张量
    #     if not isinstance(T_context, torch.Tensor):
    #         raise TypeError("self.T 的输出不是一个 PyTorch 张量")

    #     w_T_context = self.w(T_context).view(-1, self.d_flow_var, self.d_k)
    #     w_T_context_transposed = torch.transpose(w_T_context, 1, 2)

    #     z_expanded = z.unsqueeze(2)
    #     lin = torch.matmul(w_T_context_transposed, z_expanded)
    #     lin = lin.squeeze(-1)
    #     lin = lin + self.b(T_context)

    #     inner = torch.sum(self.w(T_context) * self.u(T_context))

    #     u = self.u(T_context) + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w(
    #         T_context
    #     ) / torch.sum(
    #         self.w(T_context) ** 2
    #     )  # constraint w.T * u > -1
    #     if self.act == "tanh":
    #         h_ = (
    #             lambda x: 1 / torch.cosh(x) ** 2
    #         )  # 定义了一个匿名函数（也称为 lambda 函数），它接受一个参数 x 并返回 1 / torch.cosh(x) ** 2 的值
    #     elif self.act == "leaky_relu":
    #         h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

    #     # u的形状为(-1,self.d_flow_var*self.d_k),self.h(lin)的形状是(-1,self.k)
    #     u = u.view(-1, self.d_flow_var, self.d_k)
    #     h_lin_expand = self.h(lin).unsqueeze(1)
    #     z_ = z + torch.matmul(w_T_context_transposed, h_lin_expand)
    #     log_det = torch.log(
    #         torch.abs(1 + torch.sum(self.w(T_context) * u) * h_(lin.reshape(-1)))
    #     )  # torch.sum()用于计算张量（Tensor）沿着某维度的元素之和。

    #     return z_, log_det
