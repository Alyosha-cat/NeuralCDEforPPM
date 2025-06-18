import torch
import torch.nn as nn
import torchcde

import torch
import torchdiffeqGPU
import torchsde
import warnings

# manual code reproduce to utilize GPU
def _check_compatability_per_tensor_base(control_gradient, z0):
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("X.derivative did not return a tensor with the same number of batch dimensions as z0. "
                         "X.derivative returned shape {} (meaning {} batch dimensions), whilst z0 has shape {} "
                         "(meaning {} batch dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))


def _check_compatability_per_tensor_forward(control_gradient, system, z0):
    _check_compatability_per_tensor_base(control_gradient, z0)
    if system.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(system.shape), tuple(system.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if system.size(-2) != z0.size(-1):
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-2), tuple(z0.shape), z0.size(-1)))
    if system.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as X.derivative "
                         "returned. func returned shape {} (meaning {} channels), whilst X.derivative returned shape "
                         "{} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))


def _check_compatability_per_tensor_prod(control_gradient, vector_field, z0):
    _check_compatability_per_tensor_base(control_gradient, z0)
    if vector_field.shape != z0.shape:
        raise ValueError("func.prod did not return a tensor with the same shape as z0. func.prod returned shape {} "
                         "whilst z0 has shape {}."
                         "".format(tuple(vector_field.shape), tuple(z0.shape)))


def _check_compatability(X, func, z0, t):
    if not hasattr(X, 'derivative'):
        raise ValueError("X must have a 'derivative' method.")
    control_gradient = X.derivative(t[0].detach())
    if hasattr(func, 'prod'):
        is_prod = True
        vector_field = func.prod(t[0], z0, control_gradient)
    else:
        is_prod = False
        system = func(t[0], z0)

    if isinstance(z0, torch.Tensor):
        is_tensor = True
        if not isinstance(control_gradient, torch.Tensor):
            raise ValueError("z0 is a tensor and so X.derivative must return a tensor as well.")
        if is_prod:
            if not isinstance(vector_field, torch.Tensor):
                raise ValueError("z0 is a tensor and so func.prod must return a tensor as well.")
            _check_compatability_per_tensor_prod(control_gradient, vector_field, z0)
        else:
            if not isinstance(system, torch.Tensor):
                raise ValueError("z0 is a tensor and so func must return a tensor as well.")
            _check_compatability_per_tensor_forward(control_gradient, system, z0)

    elif isinstance(z0, (tuple, list)):
        is_tensor = False
        if not isinstance(control_gradient, (tuple, list)):
            raise ValueError("z0 is a tuple/list and so X.derivative must return a tuple/list as well.")
        if len(z0) != len(control_gradient):
            raise ValueError("z0 and X.derivative(t) must be tuples of the same length.")
        if is_prod:
            if not isinstance(vector_field, (tuple, list)):
                raise ValueError("z0 is a tuple/list and so func.prod must return a tuple/list as well.")
            if len(z0) != len(vector_field):
                raise ValueError("z0 and func.prod(t, z, dXdt) must be tuples of the same length.")
            for control_gradient_, vector_Field_, z0_ in zip(control_gradient, vector_field, z0):
                if not isinstance(control_gradient_, torch.Tensor):
                    raise ValueError("X.derivative must return a tensor or tuple of tensors.")
                if not isinstance(vector_Field_, torch.Tensor):
                    raise ValueError("func.prod must return a tensor or tuple/list of tensors.")
                _check_compatability_per_tensor_prod(control_gradient_, vector_Field_, z0_)
        else:
            if not isinstance(system, (tuple, list)):
                raise ValueError("z0 is a tuple/list and so func must return a tuple/list as well.")
            if len(z0) != len(system):
                raise ValueError("z0 and func(t, z) must be tuples of the same length.")
            for control_gradient_, system_, z0_ in zip(control_gradient, system, z0):
                if not isinstance(control_gradient_, torch.Tensor):
                    raise ValueError("X.derivative must return a tensor or tuple of tensors.")
                if not isinstance(system_, torch.Tensor):
                    raise ValueError("func must return a tensor or tuple/list of tensors.")
                _check_compatability_per_tensor_forward(control_gradient_, system_, z0_)

    else:
        raise ValueError("z0 must either a tensor or a tuple/list of tensors.")

    return is_tensor, is_prod


class _VectorField(torch.nn.Module):
    def __init__(self, X, func, is_tensor, is_prod):
        super(_VectorField, self).__init__()

        self.X = X
        self.func = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)
        # cuda
        if self.is_prod:
            # out is of shape (..., hidden_channels)
            out = self.func.prod(t, z, control_gradient)
        else:
            # vector_field is of shape (..., hidden_channels, input_channels)
            vector_field = self.func(t, z)
            if self.is_tensor:
                # out is of shape (..., hidden_channels)
                # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
                out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
            else:
                out = tuple((vector_field_ @ control_gradient_.unsqueeze(-1)).squeeze(-1)
                            for vector_field_, control_gradient_ in zip(vector_field, control_gradient))
        return out

    # torchsde backend
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)


def cdeint(X, func, z0, t, adjoint=True, backend="torchdiffeq", **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcde.CubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be a callable describing the vector field f(t, z). If using `adjoint=True` (the default), then
            should be an instance of `torch.nn.Module`, to collect the parameters for the adjoint pass. Will be called
            with a scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions. If it
            has a method `prod` then that will be called to calculate the matrix-vector product f(t, z) dX_t/dt, via
            `func.prod(t, z, dXdt)`.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        backend: Either "torchdiffeq" or "torchsde". Which library to use for the solvers. Note that if using torchsde
            that the Brownian motion component is completely ignored -- so it's still reducing the CDE to an ODE --
            but it makes it possible to e.g. use an SDE solver there as the ODE/CDE solver here, if for some reason
            that's desired.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq (the most common are `rtol`, `atol`,
            `method`, `options`) or the sdeint solver of torchsde.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Note:
        Supports tupled input, i.e. z0 can be a tuple of tensors, and X.derivative and func can return tuples of tensors
        of the same length.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.odeint` or `torchsde.sdeint`.
    """

    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if adjoint:
        if "adjoint_atol" not in kwargs:
            kwargs["adjoint_atol"] = kwargs["atol"]
        if "adjoint_rtol" not in kwargs:
            kwargs["adjoint_rtol"] = kwargs["rtol"]

    is_tensor, is_prod = _check_compatability(X, func, z0, t)
    if adjoint and 'adjoint_params' not in kwargs:
        for buffer in X.buffers():
            # Compare based on id to avoid PyTorch not playing well with using `in` on tensors.
            if buffer.requires_grad:
                warnings.warn("One of the inputs to the control path X requires gradients but "
                              "`kwargs['adjoint_params']` has not been passed. This is probably a mistake: these "
                              "inputs will not receive a gradient when using the adjoint method. Either have the input "
                              "not require gradients (if that was unintended), or include it (and every other "
                              "parameter needing gradients) in `adjoint_params`. For example:\n"
                              "```\n"
                              "coeffs = ...\n"
                              "func = ...\n"
                              "X = CubicSpline(coeffs)\n"
                              "adjoint_params = tuple(func.parameters()) + (coeffs,)\n"
                              "cdeint(X=X, func=func, ..., adjoint_params=adjoint_params)\n"
                              "```")

    vector_field = _VectorField(X=X, func=func, is_tensor=is_tensor, is_prod=is_prod)
    if backend == "torchdiffeq":
        odeint = torchdiffeqGPU.odeint_adjoint if adjoint else torchdiffeqGPU.odeint
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    elif backend == "torchsde":
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        out = sdeint(sde=vector_field, y0=z0, ts=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    if is_tensor:
        batch_dims = range(1, len(out.shape) - 1)
        out = out.permute(*batch_dims, 0, -1)
    else:
        out_ = []
        for outi in out:
            batch_dims = range(1, len(outi.shape) - 1)
            outi = outi.permute(*batch_dims, 0, -1)
            out_.append(outi)
        out = tuple(out_)

    return out


# CDE Func for hidden state
class CDEFunc(nn.Module):
    def __init__(self, hidden_dim, input_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim * input_channels)

    def forward(self, t, z):
        out = self.linear(z)
        return out.view(z.size(0), z.size(1), -1)

# None use just for test RK4
def cdeint_fixed_step(X, z0, func, ts):
    """
    A fixed-step Runge-Kutta 4 (RK4) integrator for Neural CDEs.
    
    Parameters
    ----------
    X : torchcde.LinearInterpolation
        The control path object (must support .evaluate and .derivative).
    z0 : torch.Tensor
        Initial hidden state, shape [B, H].
    func : nn.Module
        The neural vector field, accepts (t, z) and returns shape [B, H, D].
    ts : torch.Tensor
        1D tensor of time points to integrate over, shape [T].
    
    Returns
    -------
    z_t : torch.Tensor
        The integrated hidden states at each time step, shape [B, T, H].
    """
    batch_size, hidden_dim = z0.shape
    T = ts.numel()
    z_t = torch.empty(batch_size, T, hidden_dim, device=z0.device, dtype=z0.dtype)

    z = z0
    z_t[:, 0] = z

    for i in range(1, T):
        t0 = ts[i - 1]
        t1 = ts[i]
        dt = t1 - t0

        # Evaluate control and derivative at midpoint
        x0 = X.evaluate(t0)              # shape: [B, D]
        dX0 = X.derivative(t0)           # shape: [B, D]

        f1 = func(t0, z)                 # [B, H, D]
        f1 = torch.bmm(f1, dX0.unsqueeze(-1)).squeeze(-1)  # [B, H]

        z1 = z + dt / 2 * f1

        f2 = func(t0 + dt / 2, z1)
        dX1 = X.derivative(t0 + dt / 2)
        f2 = torch.bmm(f2, dX1.unsqueeze(-1)).squeeze(-1)

        z2 = z + dt / 2 * f2

        f3 = func(t0 + dt / 2, z2)
        dX2 = X.derivative(t0 + dt / 2)
        f3 = torch.bmm(f3, dX2.unsqueeze(-1)).squeeze(-1)

        z3 = z + dt * f3

        f4 = func(t1, z3)
        dX3 = X.derivative(t1)
        f4 = torch.bmm(f4, dX3.unsqueeze(-1)).squeeze(-1)

        dz = (f1 + 2 * f2 + 2 * f3 + f4) / 6
        z = z + dt * dz

        z_t[:, i] = z

    return z_t


class NODE_no_context(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, num_activities: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_activities = num_activities

        self.initial = nn.Linear(input_channels, hidden_dim)
        self.func = CDEFunc(hidden_dim, input_channels)
        self.dropout = nn.Dropout(dropout)

        # Decoder: heads for suffix prediction (SuTraN style)
        self.act_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_activities)
        )
        self.ttne_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.rrt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    

    def forward(self, data, pad):
        valid = (pad==False).sum(axis=1)    # [B]
        data_fix = data.clone()
        for b in range(len(valid)):
            leng = valid[b]
            data_fix[b, leng:] = data_fix[b, leng-1]  # use last one for whole
        ts = torch.linspace(0, 1, data_fix.shape[1], device=data.device)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_fix)
        X = torchcde.CubicSpline(coeffs)

        z0 = self.initial(X.evaluate(ts[0]))
            
        # z_t = cdeint_fixed_step(X=X, z0=z0, func=self.func, ts=ts)
        z_t = cdeint(X=X, z0=z0, func=self.func, t=ts)

        # z_t is full hidden state after input
        if self.training:
            z_t = self.dropout(z_t)

            act_logits = self.act_head(z_t)         # [B, T, num_activities]
            ttne_pred = self.ttne_head(z_t)         # [B, T, 1]
            rrt_pred = self.rrt_head(z_t)           # [B, T, 1]
            return act_logits, ttne_pred, rrt_pred

        else:
            act_logits = self.act_head(z_t).argmax(-1)  # shape [B, T] → predict argmax index directly
            ttne_pred = self.ttne_head(z_t).squeeze(-1)           # shape [B, T, 1] → [B, T]
            rrt_pred = self.rrt_head(z_t[:, 0, :]).squeeze(-1)    # predict RRT from t=0 only → shape [B]

            return act_logits, ttne_pred, rrt_pred


class NODE_linear(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, num_activities: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_activities = num_activities

        self.initial = nn.Linear(input_channels, hidden_dim)
        self.func = CDEFunc(hidden_dim, input_channels)
        self.dropout = nn.Dropout(dropout)

        # Decoder: heads for suffix prediction (SuTraN style)
        self.act_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_activities)
        )
        self.ttne_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.rrt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    

    def forward(self, data, pad):
        ts = torch.linspace(0, 1, data.shape[1], device=data.device)
        coeffs = torchcde.linear_interpolation_coeffs(data)
        X = torchcde.LinearInterpolation(coeffs)

        z0 = self.initial(X.evaluate(ts[0]))
            
        # z_t = cdeint_fixed_step(X=X, z0=z0, func=self.func, ts=ts)
        z_t = cdeint(X=X, z0=z0, func=self.func, t=ts)

        # z_t is full hidden state after input
        if self.training:
            z_t = self.dropout(z_t)

            act_logits = self.act_head(z_t)         # [B, T, num_activities]
            ttne_pred = self.ttne_head(z_t)         # [B, T, 1]
            rrt_pred = self.rrt_head(z_t)           # [B, T, 1]
            return act_logits, ttne_pred, rrt_pred

        else:
            act_logits = self.act_head(z_t).argmax(-1)  # shape [B, T] → predict argmax index directly
            ttne_pred = self.ttne_head(z_t).squeeze(-1)           # shape [B, T, 1] → [B, T]
            rrt_pred = self.rrt_head(z_t[:, 0, :]).squeeze(-1)    # predict RRT from t=0 only → shape [B]

            return act_logits, ttne_pred, rrt_pred