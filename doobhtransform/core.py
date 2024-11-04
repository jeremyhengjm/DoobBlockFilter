"""
A module for the core functions of the DoobBlockFiltering package.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from doobhtransform.neuralnet import V_Network, Z_Network

from doobhtransform.utils import resampling


def construct_time_discretization(interval, M, device):
    time = torch.linspace(0.0, interval, M + 1, device=device)
    stepsizes = (interval / M) * torch.ones(M, device=device)
    return (time, stepsizes)


class model(torch.nn.Module):
    def __init__(self, state, obs, num_steps, net_config, device="cpu"):
        super(model, self).__init__()

        # latent states
        self.d = state["dim"]
        self.initial_generator = state["rinit"]
        self.b = state["drift"]
        self.sigma = state["sigma"]

        # observations
        self.p = obs["dim"]
        self.T = obs["num_obs"]
        self.interval = obs["interval"]
        self.obs_generator = obs["robs"]
        self.obs_log_density = obs["log_density"]

        # time discretization
        self.M = num_steps
        (self.time, self.stepsizes) = construct_time_discretization(
            self.interval, num_steps, device
        )

        # initialize V and Z networks
        self.V_net = V_Network(self.T, self.d, self.p, net_config["V"])
        self.V_net.to(device)

        self.Z_net = Z_Network(self.T, self.d, self.p, net_config["Z"])
        self.Z_net.to(device)

        self.training_parameters = [
            {"params": self.V_net.parameters()},
            {"params": self.Z_net.parameters()},
        ]

        # device for computations
        self.device = device

    def simulate_controlled_SDEs(
        self,
        theta,
        obs_index,
        observations,
        initial_states,
        initial_values,
        control_required=True,
    ):
        """
        Simulate controlled diffusion processes X and V for unit time using Euler-Maruyama discretization.

        Parameters
        ----------
        theta : model parameters

        obs_index : observation index (int)

        observations : observations at specified index (p)

        initial_states : initial states of X process (N, d)

        initial_values : initial values of V process (N)

        control_required : flag for whether control is applied (default=true)

        Returns
        -------
        tuple containing:
            X : X process at unit time (N, d)
            V : V process at unit time (N)
        """

        # initialize and preallocate
        t = obs_index
        Y = observations
        N = initial_states.shape[0]
        X = initial_states  # size (N, d)
        V = initial_values  # size (N)
        M = self.M
        d = self.d

        for m in range(M):
            # time step
            stepsize = self.stepsizes[m]
            s = self.time[m]

            # Brownian increment
            W = torch.sqrt(stepsize) * torch.randn(
                N, d, device=self.device
            )  # size (N, d)

            # simulate V process forwards in time
            Z = self.Z_net(t, s, X, Y)
            control = -Z.clone().detach() * control_required
            drift_V = 0.5 * torch.sum(torch.square(Z), 1) + torch.sum(
                control * Z, 1
            )  # size (N)
            euler_V = V + stepsize * drift_V  # size (N)
            V = euler_V + torch.sum(Z * W, 1)  # size (N)

            # simulate X process forwards in time
            drift_X = self.b(theta, X) + self.sigma * control
            euler_X = X + stepsize * drift_X
            X = euler_X + self.sigma * W

        return X, V

    def loss_function(
        self,
        theta,
        iteration,
        initial_required,
        minibatch,
        initial_states,
        observations,
    ):
        """
        Compute loss function at each training iteration.

        Parameters
        ----------
        theta : model parameters

        iteration : iteration index (int)

        initial_required : flag for whether initialization is required

        minibatch : minibatch size (int)

        initial_states : initial states of X process (N, d)

        observations : observation sequences (N, T, p)

        Returns
        -------
        loss : loss value
        """

        # settings
        N = minibatch
        T = self.T
        X0 = initial_states
        Y = observations

        # initialize
        X = X0
        V = self.V_net(0, X0, Y[:, 0, :])
        loss_term = torch.zeros(T, device=self.device)
        control_required = False if iteration == 0 and initial_required else True

        for t in range(T):
            # simulate X and V processes for unit time
            X, V = self.simulate_controlled_SDEs(
                theta, t, Y[:, t, :], X, V, control_required
            )

            # compute loss
            if t == (T - 1):
                loss_term[t] = torch.mean(
                    torch.square(V + self.obs_log_density(theta, X, Y[:, t, :]))
                )
            else:
                # evaluate V neural network
                V_eval = self.V_net(t + 1, X, Y[:, t + 1, :])  # check!
                loss_term[t] = torch.mean(
                    torch.square(
                        V + self.obs_log_density(theta, X, Y[:, t, :]) - V_eval
                    )
                )

            # update initial values
            V = V_eval

        loss = torch.sum(loss_term)

        return loss

    def train(self, theta, optim_config):
        """
        Train approximations iteratively.

        Parameters
        ----------
        theta : model parameters

        optim_config : configuration of optimizer

        Returns
        -------
        loss : value of loss function during learning (num_iterations)
        """

        # optimization configuration
        minibatch = optim_config["minibatch"]
        num_iterations = optim_config["num_iterations"]
        learning_rate = optim_config["learning_rate"]
        weight_decay = optim_config["weight_decay"]
        initial_required = optim_config["initial_required"]
        optimizer = torch.optim.AdamW(
            self.training_parameters, lr=learning_rate, weight_decay=weight_decay
        )

        # optimization
        loss_values = torch.zeros(num_iterations, device=self.device)

        for i in tqdm(range(num_iterations)):
            # simulate initial states X0 and observations sequence Y
            initial_states = self.initial_generator(minibatch)
            observations = self.obs_generator(minibatch)

            # run forward pass and compute loss
            loss = self.loss_function(
                theta, i, initial_required, minibatch, initial_states, observations
            )

            # backpropagation
            loss.backward()

            # optimization step and zero gradient
            optimizer.step()
            optimizer.zero_grad()

            # store loss
            current_loss = loss.item()
            loss_values[i] = current_loss
            if (i == 0) or ((i + 1) % 100 == 0):
                print("Optimization iteration:", i + 1, "Loss:", current_loss)

    def simulate_uncontrolled_SMC(
        self,
        theta,
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
        theta : model parameters

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
        T = self.T
        d = self.d
        M = self.M
        Y = observations
        X = initial_states.repeat(N, 1)
        if full_path:
            states = torch.zeros(N, T * M + 1, d, device=self.device)
        else:
            states = torch.zeros(N, T + 1, d, device=self.device)
        states[:, 0, :] = X
        ess = torch.zeros(T + 1, device=self.device)
        ess[0] = N
        log_norm_const = torch.zeros(T + 1, device=self.device)
        log_ratio_norm_const = torch.tensor(0.0, device=self.device)

        # simulate X process
        for t in range(T):
            # unit time interval
            for m in range(M):
                # time step
                stepsize = self.stepsizes[m]
                s = self.time[m]

                # Brownian increment
                W = torch.sqrt(stepsize) * torch.randn(
                    N, d, device=self.device
                )  # size (N, d)

                # simulate X process forwards in time
                euler_X = X + stepsize * self.b(theta, X)
                X = euler_X + self.sigma * W
                if full_path:
                    index = t * M + m + 1
                    states[:, index, :] = X

            # compute and normalize weights, compute ESS and normalizing constant
            log_weights = self.obs_log_density(theta, X, Y[t, :])
            max_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_weights)
            normalized_weights = weights / torch.sum(weights)
            ess[t + 1] = 1.0 / torch.sum(normalized_weights**2)
            log_ratio_norm_const = (
                log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
            )
            log_norm_const[t + 1] = log_ratio_norm_const

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
        self,
        theta,
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
        theta : model parameters

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
        T = self.T
        d = self.d
        M = self.M
        Y = observations
        X = initial_states.repeat(N, 1)
        with torch.no_grad():
            V = self.V_net(0, X, Y[0, :])
        if full_path:
            states = torch.zeros(N, T * M + 1, d, device=self.device)
        else:
            states = torch.zeros(N, T + 1, d, device=self.device)
        states[:, 0, :] = X
        ess = torch.zeros(T + 1, device=self.device)
        ess[0] = N
        log_norm_const = torch.zeros(T + 1, device=self.device)
        log_ratio_norm_const = -V[0]  # may need to generalize this

        # simulate X process
        for t in range(T):
            # unit time interval
            for m in range(M):
                # time step
                stepsize = self.stepsizes[m]
                s = self.time[m]

                # Brownian increment
                W = torch.sqrt(stepsize) * torch.randn(
                    N, d, device=self.device
                )  # size (N, d)

                # simulate V process forwards in time
                with torch.no_grad():
                    Z = self.Z_net(t, s, X, Y[t, :])
                control = -Z.clone()
                drift_V = -0.5 * torch.sum(torch.square(Z), 1)  # size (N)
                euler_V = V + stepsize * drift_V  # size (N)
                V = euler_V + torch.sum(Z * W, 1)  # size (N)

                # simulate X process forwards in time
                drift_X = self.b(theta, X) + self.sigma * control
                euler_X = X + stepsize * drift_X
                X = euler_X + self.sigma * W
                if full_path:
                    index = t * M + m + 1
                    states[:, index, :] = X

            # compute log-weights
            if t == (T - 1):
                log_weights = V + self.obs_log_density(theta, X, Y[t, :])
            else:
                # evaluate V neural network
                with torch.no_grad():
                    V_eval = self.V_net(t + 1, X, Y[t + 1, :])
                log_weights = V + self.obs_log_density(theta, X, Y[t, :]) - V_eval

            # normalize weights, compute ESS and normalizing constant
            max_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_weights)
            normalized_weights = weights / torch.sum(weights)
            ess[t + 1] = 1.0 / torch.sum(normalized_weights**2)
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
