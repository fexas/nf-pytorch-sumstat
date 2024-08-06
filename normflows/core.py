import torch
import torch.nn as nn
import numpy as np

from . import distributions
from . import utils
from torch.autograd.functional import jacobian


import warnings


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            print(
                "core.py-shape of z, log_det,log_q:",
                z.shape,
                log_det.shape,
                log_q.shape,
            )  # debug
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class ConditionalNormalizingFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """

    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def linear_mmd(self, num_samples=1, x=None, context=None):
        """
        modify in 7.3
        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var

        Returns:
          Estimate of linear MMD over batch
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det

        delta = x - z
        delta_T = delta.T
        product = torch.matmul(delta, delta_T)
        product = product.type(torch.float32)
        loss = torch.mean(product)
        return loss

    def test_mmd(self, num_samples=1, x=None, context=None, num_z=1):
        """
        modify in 7.3
        modify in 7.4: add parameter_num_z
        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context
          num_z:  number of flow variable to generate (estimate expectation of kernel)

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var
          z_list: batch_size * d_flow_var * num_z

        Returns:
          Estimate of linear MMD over batch
        """

        def all_combinations(tensor):
            num_elements = len(tensor)
            combinations = []

            for i in range(num_elements):
                for j in range(num_elements):
                    if j != i:
                        combinations.append(torch.tensor([tensor[i], tensor[j]]))

            return torch.stack(combinations)

        def inner_kernel_function(input_tensor):
            # 检查输入tensor的形状
            assert input_tensor.shape[1] == 2, "Input tensor must have 2 columns."

            # 计算每行两个元素之差的平方
            diff_squared = (input_tensor[:, 0] - input_tensor[:, 1]) ** 2

            # 计算指数并返回结果
            return torch.exp(-diff_squared / 2)  # shape: num_z

        # 创建一个形状为(num_z, 2)的示例tensor

        z_list = []

        for r in range(num_z):
            z, log_q_ = self.q0(num_samples, context=context)
            log_q = torch.zeros_like(log_q_)
            log_q += log_q_
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det
            z_list.append(z)

        z_list_tensor = torch.stack(
            z_list, dim=-1
        )  # shape: batch_size * d_flow_var * num_z
        # 从张量z中提取batch_size和d_flow_var
        _batch_size = z.shape[0]
        _d_flow_var = z.shape[1]

        # first term

        first_term_tensor = torch.zeros_like(
            z
        )  # shape: batch_size * d_flow_var (用来存每个位置kernel_sum的结果)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                # 使用torch.combinations获取两两不同元素的组合，考虑前后顺序
                combinations = all_combinations(
                    z_element
                )  # 出来应该是一个tensor,形状是(num_z,2)
                # 将组合转换为tensor
                combinations_tensor = torch.stack(list(combinations))
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()  # 类型依旧是tensor
                first_term_tensor[s, t] = mean_of_kernel

        first_term = first_term_tensor.sum() / _batch_size

        # second Term
        second_term_tensor = torch.zeros_like(z)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                real_theta = x[s, t]  # 会取出一个元素，torch.Size([])
                # 将 real_theta 重复 num_z 次
                repeated_real_theta = torch.repeat_interleave(real_theta, num_z)
                # 将 repeated_real_theta 和 z_element 堆叠成一个二维张量
                combinations_tensor = torch.stack(
                    (repeated_real_theta, z_element), dim=1
                )
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()
                second_term_tensor[s, t] = mean_of_kernel

        second_term = -2 * second_term_tensor.sum() / _batch_size

        loss = first_term + second_term
        return loss


class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model, providing the
    class to be conditioned on only to the base distribution,
    as done e.g. in [Glow](https://arxiv.org/abs/1807.03039)
    """

    def __init__(self, q0, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x, y):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
         param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of list of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        self.q0 = nn.ModuleList(q0)
        self.num_levels = len(self.q0)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.transform = transform
        self.class_cond = class_cond

    def forward_kld(self, x, y=None):
        """Estimates forward KL divergence, see see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of targets, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x, y=None):
        """Get negative log-likelihood for maximum likelihood training

        Args:
          x: Batch of data
          y: Batch of targets, if applicable

        Returns:
            Negative log-likelihood of the batch
        """
        return -self.log_prob(x, y)

    def forward_and_log_det(self, z):
        """Get observed variable x from list of latent variables z

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z[0].device)
        for i in range(len(self.q0)):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_)
                log_det += log_det_
        if self.transform is not None:
            z_, log_det_ = self.transform(z_)
            log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        if self.transform is not None:
            x, log_det_ = self.transform.inverse(x)
            log_det += log_det_
        z = [None] * len(self.q0)
        for i in range(len(self.q0) - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def sample(self, num_samples=1, y=None, temperature=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)
        for i in range(len(self.q0)):
            if self.class_cond:
                z_, log_q_ = self.q0[i](num_samples, y)
            else:
                z_, log_q_ = self.q0[i](num_samples)
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(len(self.q0) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            if self.class_cond:
                log_q += self.q0[i].log_prob(z_, y)
            else:
                log_q += self.q0[i].log_prob(z_)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, "temperature"):
                q0.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not "
                    "support temperature annealed sampling"
                )

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)


class NormalizingFlowVAE(nn.Module):
    """
    VAE using normalizing flows to express approximate distribution
    """

    def __init__(self, prior, q0=distributions.Dirac(), flows=None, decoder=None):
        """Constructor of normalizing flow model

        Args:
          prior: Prior distribution of te VAE, i.e. Gaussian
          decoder: Optional decoder
          flows: Flows to transform output of base encoder
          q0: Base Encoder
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0

    def forward(self, x, num_samples=1):
        """Takes data batch, samples num_samples for each data point from base distribution

        Args:
          x: data batch
          num_samples: number of samples to draw for each data point

        Returns:
          latent variables for each batch and sample, log_q, and log_p
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_prob(x, z)
        # Separate batch and sample dimension again
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        return z, log_q, log_p


# 施工中-7.9


class ConditionalNormalizingFlowSumStat(nn.Module):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers

    + SumStat
    """

    def __init__(self, q0, flows, p=None, MLP_sumstat=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p
        self.MLP_sumstat = MLP_sumstat

    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """

        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        # 在之前加代码处理context
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """

        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def linear_mmd(self, num_samples=1, x=None, context=None):
        """
        modify in 7.3
        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var

        Returns:
          Estimate of linear MMD over batch
        """
        # 在之前加代码处理context
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det

        delta = x - z
        delta_T = delta.T
        product = torch.matmul(delta, delta_T)
        product = product.type(torch.float32)
        loss = torch.mean(product)
        return loss

    def test_mmd(self, num_samples=1, x=None, context=None, num_z=1):
        """
        modify in 7.3
        modify in 7.4: add parameter_num_z
        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context
          num_z:  number of flow variable to generate (estimate expectation of kernel)

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var
          z_list: batch_size * d_flow_var * num_z

        Returns:
          Estimate of linear MMD over batch
        """
        # 在之前加代码处理context
        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        def all_combinations(tensor):
            num_elements = len(tensor)
            combinations = []

            for i in range(num_elements):
                for j in range(num_elements):
                    if j != i:
                        combinations.append(torch.tensor([tensor[i], tensor[j]]))

            return torch.stack(combinations)

        def inner_kernel_function(input_tensor):
            # 检查输入tensor的形状
            assert input_tensor.shape[1] == 2, "Input tensor must have 2 columns."

            # 计算每行两个元素之差的平方
            diff_squared = (input_tensor[:, 0] - input_tensor[:, 1]) ** 2

            # 计算指数并返回结果
            return torch.exp(-diff_squared / 2)  # shape: num_z

        # 创建一个形状为(num_z, 2)的示例tensor

        z_list = []

        for r in range(num_z):
            z, log_q_ = self.q0(num_samples, context=context)
            log_q = torch.zeros_like(log_q_)
            log_q += log_q_
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det
            z_list.append(z)

        z_list_tensor = torch.stack(
            z_list, dim=-1
        )  # shape: batch_size * d_flow_var * num_z
        # 从张量z中提取batch_size和d_flow_var
        _batch_size = z.shape[0]
        _d_flow_var = z.shape[1]

        # first term

        first_term_tensor = torch.zeros_like(
            z
        )  # shape: batch_size * d_flow_var (用来存每个位置kernel_sum的结果)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                # 使用torch.combinations获取两两不同元素的组合，考虑前后顺序
                combinations = all_combinations(
                    z_element
                )  # 出来应该是一个tensor,形状是(num_z,2)
                # 将组合转换为tensor
                combinations_tensor = torch.stack(list(combinations))
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()  # 类型依旧是tensor
                first_term_tensor[s, t] = mean_of_kernel

        first_term = first_term_tensor.sum() / _batch_size

        # second Term
        second_term_tensor = torch.zeros_like(z)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                real_theta = x[s, t]  # 会取出一个元素，torch.Size([])
                # 将 real_theta 重复 num_z 次
                repeated_real_theta = torch.repeat_interleave(real_theta, num_z)
                # 将 repeated_real_theta 和 z_element 堆叠成一个二维张量
                combinations_tensor = torch.stack(
                    (repeated_real_theta, z_element), dim=1
                )
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()
                second_term_tensor[s, t] = mean_of_kernel

        second_term = -2 * second_term_tensor.sum() / _batch_size

        loss = first_term + second_term
        return loss

    def test_mmd_with_one_penalty(
        self,
        num_samples=1,
        x=None,
        context=None,
        num_z=1,
        weighted_matrix=None,
        first_lambda=1,
    ):
        """
        modify in 7.3
        modify in 7.4: add parameter_num_z
        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context
          num_z:  number of flow variable to generate (estimate expectation of kernel)

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var
          z_list: batch_size * d_flow_var * num_z

        Returns:
          Estimate of linear MMD over batch
        """
        if weighted_matrix is None:
            warnings.warn("weighted_matrix is None.")

        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)

        def all_combinations(tensor):
            num_elements = len(tensor)
            combinations = []

            for i in range(num_elements):
                for j in range(num_elements):
                    if j != i:
                        combinations.append(torch.tensor([tensor[i], tensor[j]]))

            return torch.stack(combinations)

        def inner_kernel_function(input_tensor):
            # 检查输入tensor的形状
            assert input_tensor.shape[1] == 2, "Input tensor must have 2 columns."

            # 计算每行两个元素之差的平方
            diff_squared = (input_tensor[:, 0] - input_tensor[:, 1]) ** 2

            # 计算指数并返回结果
            return torch.exp(-diff_squared / 2)  # shape: num_z

        # 创建一个形状为(num_z, 2)的示例tensor

        z_list = []

        for r in range(num_z):
            z, log_q_ = self.q0(num_samples, context=context)
            log_q = torch.zeros_like(log_q_)
            log_q += log_q_
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det
            z_list.append(z)

        z_list_tensor = torch.stack(
            z_list, dim=-1
        )  # shape: batch_size * d_flow_var * num_z
        # 从张量z中提取batch_size和d_flow_var
        _batch_size = z.shape[0]
        _d_flow_var = z.shape[1]

        # first term

        first_term_tensor = torch.zeros_like(
            z
        )  # shape: batch_size * d_flow_var (用来存每个位置kernel_sum的结果)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                # 使用torch.combinations获取两两不同元素的组合，考虑前后顺序
                combinations = all_combinations(
                    z_element
                )  # 出来应该是一个tensor,形状是(num_z,2)
                # 将组合转换为tensor
                combinations_tensor = torch.stack(list(combinations))
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()  # 类型依旧是tensor
                first_term_tensor[s, t] = mean_of_kernel

        first_term = first_term_tensor.sum() / _batch_size

        # second Term
        second_term_tensor = torch.zeros_like(z)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                real_theta = x[s, t]  # 会取出一个元素，torch.Size([])
                # 将 real_theta 重复 num_z 次
                repeated_real_theta = torch.repeat_interleave(real_theta, num_z)
                # 将 repeated_real_theta 和 z_element 堆叠成一个二维张量
                combinations_tensor = torch.stack(
                    (repeated_real_theta, z_element), dim=1
                )
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()
                second_term_tensor[s, t] = mean_of_kernel

        second_term = -2 * second_term_tensor.sum() / _batch_size

        # 1st penalty

        def compute_pairwise_square_distances(tensor):
            batch_size, d = tensor.shape
            theta_distance_matrix = torch.zeros(
                batch_size, batch_size, device=tensor.device
            )

            # 使用广播计算所有两两之间的差值
            diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)

            # 计算平方差值
            square_diff = diff**2

            # 计算每对样本之间的距离，并填充到矩阵中
            theta_distance_matrix = torch.sum(square_diff, dim=-1).triu()

            return theta_distance_matrix

        distance_of_T_context = compute_pairwise_square_distances(
            context
        )  # (batch_size,batch_size)

        if (
            weighted_matrix is not None
            and distance_of_T_context.shape != weighted_matrix.shape
        ):
            warnings.warn(
                "The shape of distance_of_T_context and weighted_matrix are different."
            )

        weigthed_distance_of_T_context = weighted_matrix * distance_of_T_context

        first_penalty = first_lambda * torch.sum(weigthed_distance_of_T_context)

        loss = first_term + second_term + first_penalty
        return loss

    def test_mmd_with_two_penalty(
        self,
        num_samples=1,
        x=None,
        context=None,
        num_z=1,
        weighted_matrix=None,
        first_lambda=1,
        second_lambda=1,
        num_of_estimate_expectation=1,
    ):
        """
        modify in 7.3
        modify in 7.4: add parameter_num_z
        modification in 7.10 : add second penalty

        Estimate linear MMD:
        K(f(x), f(y)) = f(x)^Tf(y)
        h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
        = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]

        Args:
          num_samples: batch_size
          x: Batch sampled from target distribution
          context: Batch of conditions/context
          num_z:  number of flow variable to generate (estimate expectation of kernel)

        Parameter Shape:
          z: batch_size * d_flow_var
          x: batch_size * d_flow_var
          z_list: batch_size * d_flow_var * num_z

        Returns:
          Estimate of linear MMD over batch
        """
        if weighted_matrix is None:
            warnings.warn("weighted_matrix is None.")

        # 在之前加代码处理context
        temp_batch_size, temp_n, temp_d_context = context.shape
        context_reshaped = context.reshape(-1, temp_n * temp_d_context)
        context = self.MLP_sumstat(context_reshaped)  # shape: (batch_size,d_compress)
        # 设置 context 的 requires_grad 属性为 True -- 7.10
        # context.requires_grad_(True)

        def all_combinations(tensor):
            num_elements = len(tensor)
            combinations = []

            for i in range(num_elements):
                for j in range(num_elements):
                    if j != i:
                        combinations.append(torch.tensor([tensor[i], tensor[j]]))

            return torch.stack(combinations)

        def inner_kernel_function(input_tensor):
            # 检查输入tensor的形状
            assert input_tensor.shape[1] == 2, "Input tensor must have 2 columns."

            # 计算每行两个元素之差的平方
            diff_squared = (input_tensor[:, 0] - input_tensor[:, 1]) ** 2

            # 计算指数并返回结果
            return torch.exp(-diff_squared / 2)  # shape: num_z

        # 创建一个形状为(num_z, 2)的示例tensor

        z_list = []

        for r in range(num_z):
            z, log_q_ = self.q0(num_samples, context=context)
            log_q = torch.zeros_like(log_q_)
            log_q += log_q_
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det
            z_list.append(z)

        z_list_tensor = torch.stack(
            z_list, dim=-1
        )  # shape: batch_size , d_flow_var , num_z
        # 从张量z中提取batch_size和d_flow_var
        _batch_size = z.shape[0]
        _d_flow_var = z.shape[1]

        # first term

        first_term_tensor = torch.zeros_like(
            z
        )  # shape: batch_size * d_flow_var (用来存每个位置kernel_sum的结果)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                # 使用torch.combinations获取两两不同元素的组合，考虑前后顺序
                combinations = all_combinations(
                    z_element
                )  # 出来应该是一个tensor,形状是(num_z,2)
                # 将组合转换为tensor
                combinations_tensor = torch.stack(list(combinations))
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()  # 类型依旧是tensor
                first_term_tensor[s, t] = mean_of_kernel

        first_term = first_term_tensor.sum() / _batch_size

        # second Term
        second_term_tensor = torch.zeros_like(z)

        for s in range(_batch_size):
            for t in range(_d_flow_var):
                z_element = z_list_tensor[
                    s, t
                ]  # 会提取出一个长为num_z的一维tensor,是 \hat{\theta}_{st}^k k = 1,..,num_z
                real_theta = x[s, t]  # 会取出一个元素，torch.Size([])
                # 将 real_theta 重复 num_z 次
                repeated_real_theta = torch.repeat_interleave(real_theta, num_z)
                # 将 repeated_real_theta 和 z_element 堆叠成一个二维张量
                combinations_tensor = torch.stack(
                    (repeated_real_theta, z_element), dim=1
                )
                kernel_of_combination = inner_kernel_function(combinations_tensor)
                mean_of_kernel = kernel_of_combination.mean()
                second_term_tensor[s, t] = mean_of_kernel

        second_term = -2 * second_term_tensor.sum() / _batch_size

        # 1st penalty

        def compute_pairwise_square_distances(tensor):
            batch_size, d = tensor.shape
            theta_distance_matrix = torch.zeros(
                batch_size, batch_size, device=tensor.device
            )

            # 使用广播计算所有两两之间的差值
            diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)

            # 计算平方差值
            square_diff = diff**2

            # 计算每对样本之间的距离，并填充到矩阵中
            theta_distance_matrix = torch.sum(square_diff, dim=-1).triu()

            return theta_distance_matrix

        distance_of_T_context = compute_pairwise_square_distances(
            context
        )  # (batch_size,batch_size)

        if (
            weighted_matrix is not None
            and distance_of_T_context.shape != weighted_matrix.shape
        ):
            warnings.warn(
                "The shape of distance_of_T_context and weighted_matrix are different."
            )

        weigthed_distance_of_T_context = weighted_matrix * distance_of_T_context

        first_penalty = first_lambda * torch.sum(weigthed_distance_of_T_context)

        # 2nd penalty
        # 可以用z_list_tensor做 gradient的估计
        # z_list_tensor 的形状是（batch_size , d_flow_var , num_z）
        # 那就是 第i个batch的(d_flow_var,num_z) 的列向量分别关于 T(x_{1:n}^i)求导
        # T(x_{1:n})的形状是(batch_size,d_compress)
        # 所以是第i个batch关于T(x_{1:n})第i行向量求导，每列关于行求导形成的jacobian矩阵形状应该是（d_flow_var,d_compress）（需要求其模的平方）, 有num_z个这样的矩阵（对模的平方求和）
        # 最终关于i求和

        # 要么就一个一个算
        # shape of context : (batch_size, d_compress)
        # 为了计算jacobian矩阵，应该还要写一个apply_flow_function

        def apply_flow(temp_context):
            z, _ = self.q0(1, context=temp_context)
            for flow in self.flows:
                z, _ = flow(z, context=temp_context)
            # 此时z的形状是(batch_size,d_flow_var)--特别的batch_size=1
            return z.squeeze(0)

        second_penalty = 0

        for i in range(_batch_size):
            temp_context = context[i, :].unsqueeze(
                0
            )  # 提取T(x_{1:n})的第i行 -- shape:（1,d_compress）
            temp_context.requires_grad_(True)
            # 分开抽取z次
            for j in range(num_of_estimate_expectation):
                jacobian_matrix = jacobian(
                    apply_flow, temp_context
                )  # 形状应该是（d_flow_var,1,d_compress）
                jacobian_matrix = jacobian_matrix.squeeze(
                    1
                )  # 形状变为 （d_flow_var,d_compress）
                second_penalty += torch.sum(jacobian_matrix**2)

        second_penalty = second_penalty / (num_of_estimate_expectation * _batch_size)

        second_penalty = second_penalty * second_lambda

        loss = first_term + second_term + first_penalty + second_penalty

        return loss


# 施工中 -- 7.18
# 主要替换用flows产出的部分
# 定义需要输入的参数
