"""
A module for block particle filtering.
"""

import torch
from tqdm import tqdm
from doobhtransform.utils import resampling


def _simulate_controlled_SDEs(
    model,
    obs_index,
    observations,
    initial_states,
    initial_values,
):

    t = obs_index
    Y = observations  # (T, p)
    N = initial_states.shape[0]
    X = initial_states  # size (N, d)
    V = initial_values  # size (N)
    M = model.M
    d = model.d

    for m in range(M):
        # time step
        stepsize = model.stepsizes[m]
        s = model.time[m]

        # Brownian increment
        W = torch.sqrt(stepsize) * torch.randn(N, d, device=model.device)  # size (N, d)

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

    return X, V


def _block_proposal_step(
    model,
    initial_states,
    observations,
):

    # initialize and preallocate
    N = initial_states.shape[0]
    T = model.T  # lag or block size
    d = model.d
    Y = observations  # block of observations of size (T, p)
    X = initial_states  # (N, d)
    with torch.no_grad():
        V = model.V_net(0, X, Y)
    states = torch.zeros(N, T + 1, d, device=model.device)
    states[:, 0, :] = X
    log_weights = -V

    for t in range(T):
        # simulate X and V processes for unit time
        X, V = _simulate_controlled_SDEs(model, t, Y, X, V)

        # compute log-weights
        if t == (T - 1):
            log_incremental_weights = V + model.obs_log_density(X, Y[t, :])
        else:
            # evaluate V neural network
            with torch.no_grad():
                V_eval = model.V_net(t + 1, X, Y)
            log_incremental_weights = V + model.obs_log_density(X, Y[t, :]) - V_eval
        log_weights += log_incremental_weights

        # compute log-auxiliary weights
        if t == 0:
            log_auxiliary_weights = -V_eval
        else:
            log_auxiliary_weights += log_incremental_weights

        # update initial values
        V = V_eval

        # store states
        states[:, t + 1, :] = X

    return states, log_weights, log_auxiliary_weights


def simulate_block_SMC(
    model,
    initial_states,
    observations,
    num_samples,
    resample=True,
):
    """
    Simulate block sequential Monte Carlo.

    Parameters
    ----------
    model : model object

    initial_states : initial states of X process (d)

    observations : sequence of observations to be filtered (T_obs, p)

    num_samples : sample size (int)

    resample : if resampling is required (bool)

    full_path : if full path of X is required (bool)

    Returns
    -------
    states : X process at unit times (N, T_obs+1, d)

    ess : effective sample sizes at unit times (T_obs+1)

    log_norm_const : log-normalizing constant estimates (T_obs+1)
    """

    # setup
    N = num_samples
    T = model.T  # lag or block size
    T_obs = observations.shape[0]  # number of observations to filter
    assert T_obs >= T, "Block size should not be larger than number of observations"
    d = model.d
    X0 = initial_states.repeat(N, 1)
    states = [X0]
    ancestries = []
    ess = torch.zeros(T_obs, device=model.device)
    log_norm_const = torch.zeros(T_obs, device=model.device)

    # initialize
    Y = observations[:T]
    current_states, log_weights, log_auxiliary_weights = _block_proposal_step(
        model, X0, Y
    )
    states.append(current_states)
    # states[:, 1, :] = current_states[:, 1, :]
    max_log_weights = torch.max(log_weights)
    weights = torch.exp(log_weights - max_log_weights)
    normalized_weights = weights / torch.sum(weights)
    ess[:T] = 1.0 / torch.sum(normalized_weights**2)
    log_ratio_norm_const = torch.log(torch.mean(weights)) + max_log_weights
    log_norm_const[:T] = log_ratio_norm_const

    # resampling
    if resample:
        ancestors = resampling(normalized_weights, N)
        ancestries.append(ancestors)
        current_states = current_states[ancestors, :, :]
        log_auxiliary_weights = log_auxiliary_weights[ancestors]
        normalized_weights = torch.ones(N) * 1.0 / N

    # loop over remaining observations
    for t in range(T_obs - T):
        # block proposal
        X0 = current_states[:, 1, :]
        Y = observations[(t + 1) : (t + 1 + T)]
        new_states, new_log_weights, new_log_auxiliary_weights = _block_proposal_step(
            model, X0, Y
        )
        states.append(new_states)
        # else:
        #     if t == (T_obs - T - 1):
        #         states[:, t + 2, :]
        #     else:
        #         states[:, -T:, :] = new_states[:, 1:, :]

        # importance weighting
        log_weights = (
            new_log_weights - log_auxiliary_weights + torch.log(N * normalized_weights)
        )
        max_log_weights = torch.max(log_weights)
        weights = torch.exp(log_weights - max_log_weights)
        normalized_weights = weights / torch.sum(weights)
        ess[T + t] = 1.0 / torch.sum(normalized_weights**2)
        log_ratio_norm_const = (
            log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
        )
        log_norm_const[T + t] = log_ratio_norm_const

        # resampling
        if resample:
            ancestors = resampling(normalized_weights, N)
            ancestries.append(ancestors)
            new_states = new_states[ancestors, :, :]
            new_log_auxiliary_weights = new_log_auxiliary_weights[ancestors]
            normalized_weights = torch.ones(N) * 1.0 / N

        # update variables
        current_states = new_states
        log_auxiliary_weights = new_log_auxiliary_weights

    # output
    output = {
        "states": states,
        "ess": ess,
        "log_norm_const": log_norm_const,
        "log_ratio_norm_const": log_ratio_norm_const,
    }

    return output
