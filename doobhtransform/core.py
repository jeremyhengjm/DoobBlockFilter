"""
A module for Doob's h-transform.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from doobhtransform.neuralnet import V_Network, Z_Network


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

        observations : observations at specified index (N, T, p)

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
        V = self.V_net(0, X0, Y)
        loss_term = torch.zeros(T, device=self.device)
        control_required = False if iteration == 0 and initial_required else True

        for t in range(T):
            # simulate X and V processes for unit time
            X, V = self.simulate_controlled_SDEs(theta, t, Y, X, V, control_required)

            # compute loss
            if t == (T - 1):
                loss_term[t] = torch.mean(
                    torch.square(V + self.obs_log_density(theta, X, Y[:, t, :]))
                )
            else:
                # evaluate V neural network
                V_eval = self.V_net(t + 1, X, Y)
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
