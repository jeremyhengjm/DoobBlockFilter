"""
A module for particle filtering.
"""

import torch
from tqdm import tqdm
from doobhtransform.utils import resampling


def simulate_uncontrolled_SMC(
    model,
    initial_states,
    observations,
    num_samples,
    resample=False,
    full_path=False,
):
    """
    Simulate uncontrolled sequential Monte Carlo.

    Parameters
    ----------
    model : model object

    initial_states : initial states of X process (d)

    observations : sequence of observations to be filtered (T, p)

    num_samples : sample size (int)

    resample : if resampling is required (bool)

    full_path : if full path of X is required (bool)

    Returns
    -------
    states : X process at unit times (N, T+1, d)

    ess : effective sample sizes at unit times (T+1)

    log_norm_const : log-normalizing constant estimates (T+1)
    """

    # initialize and preallocate
    N = num_samples
    T = observations.shape[0]
    d = model.d
    M = model.M
    Y = observations
    X = initial_states.repeat(N, 1)
    if full_path:
        states = torch.zeros(N, T * M + 1, d, device=model.device)
    else:
        states = torch.zeros(N, T + 1, d, device=model.device)
    states[:, 0, :] = X
    ess = torch.zeros(T, device=model.device)
    log_norm_const = torch.zeros(T, device=model.device)
    log_ratio_norm_const = torch.tensor(0.0, device=model.device)

    # simulate X process
    for t in range(T):
        # unit time interval
        for m in range(M):
            # time step
            stepsize = model.stepsizes[m]
            s = model.time[m]

            # Brownian increment
            W = torch.sqrt(stepsize) * torch.randn(
                N, d, device=model.device
            )  # size (N, d)

            # simulate X process forwards in time
            euler_X = X + stepsize * model.b(X)
            X = euler_X + model.sigma * W
            if full_path:
                index = t * M + m + 1
                states[:, index, :] = X

        # compute and normalize weights, compute ESS and normalizing constant
        log_weights = model.obs_log_density(X, Y[t, :])
        max_log_weights = torch.max(log_weights)
        weights = torch.exp(log_weights - max_log_weights)
        normalized_weights = weights / torch.sum(weights)
        ess[t] = 1.0 / torch.sum(normalized_weights**2)
        log_ratio_norm_const = (
            log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
        )
        log_norm_const[t] = log_ratio_norm_const

        # resampling
        if resample:
            ancestors = resampling(normalized_weights, N)
            X = X[ancestors, :]

        # store states
        if full_path:
            index_start = t * M + 1
            index_end = t * M + M + 1
            states[:, index_start:index_end, :] = states[
                ancestors, index_start:index_end, :
            ]
        else:
            states[:, t + 1, :] = X

    return states, ess, log_norm_const, log_ratio_norm_const


def simulate_controlled_SMC(
    model,
    initial_states,
    observations,
    num_samples,
    resample=False,
    full_path=False,
):
    """
    Simulate controlled sequential Monte Carlo.

    Parameters
    ----------
    model : model object

    initial_states : initial states of X process (d)

    observations : sequence of observations to be filtered (T, p)

    num_samples : sample size (int)

    resample : if resampling is required (bool)

    full_path : if full path of X is required (bool)

    Returns
    -------
    states : X process at unit times (N, T+1, d)

    ess : effective sample sizes at unit times (T+1)

    log_norm_const : log-normalizing constant estimates (T+1)
    """

    # initialize and preallocate
    N = num_samples
    T = model.T
    d = model.d
    M = model.M
    Y = observations
    X = initial_states.repeat(N, 1)
    with torch.no_grad():
        V = model.V_net(0, X, Y)
    if full_path:
        states = torch.zeros(N, T * M + 1, d, device=model.device)
    else:
        states = torch.zeros(N, T + 1, d, device=model.device)
    states[:, 0, :] = X
    ess = torch.zeros(T, device=model.device)
    log_norm_const = torch.zeros(T + 1, device=model.device)
    log_norm_const[0] = -V[0]
    log_ratio_norm_const = -V[0]  # may need to generalize this

    # simulate X process
    for t in range(T):
        # unit time interval
        for m in range(M):
            # time step
            stepsize = model.stepsizes[m]
            s = model.time[m]

            # Brownian increment
            W = torch.sqrt(stepsize) * torch.randn(
                N, d, device=model.device
            )  # size (N, d)

            # simulate V process forwards in time
            with torch.no_grad():
                Z = model.Z_net(t, s, X, Y)
            control = -Z.clone()
            drift_V = -0.5 * torch.sum(torch.square(Z), 1)  # size (N)
            euler_V = V + stepsize * drift_V  # size (N)
            V = euler_V + torch.sum(Z * W, 1)  # size (N)

            # simulate X process forwards in time
            drift_X = model.b(X) + model.sigma * control
            euler_X = X + stepsize * drift_X
            X = euler_X + model.sigma * W
            if full_path:
                index = t * M + m + 1
                states[:, index, :] = X

        # compute log-weights
        if t == (T - 1):
            log_weights = V + model.obs_log_density(X, Y[t, :])
        else:
            # evaluate V neural network
            with torch.no_grad():
                V_eval = model.V_net(t + 1, X, Y)
            log_weights = V + model.obs_log_density(X, Y[t, :]) - V_eval

        # normalize weights, compute ESS and normalizing constant
        max_log_weights = torch.max(log_weights)
        weights = torch.exp(log_weights - max_log_weights)
        normalized_weights = weights / torch.sum(weights)
        ess[t] = 1.0 / torch.sum(normalized_weights**2)
        log_ratio_norm_const = (
            log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
        )
        log_norm_const[t + 1] = log_ratio_norm_const

        # resampling
        if resample:
            ancestors = resampling(normalized_weights, N)
            X = X[ancestors, :]
            V_eval = V_eval[ancestors]

        # update initial values
        V = V_eval

        # store states
        if full_path:
            index_start = t * M + 1
            index_end = t * M + M + 1
            states[:, index_start:index_end, :] = states[
                ancestors, index_start:index_end, :
            ]
        else:
            states[:, t + 1, :] = X

    return states, ess, log_norm_const, log_ratio_norm_const
