# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from src.dHSIC import make_elbo_hsic
from src.vae.utils.vae_plots import plot_llk

pyro.set_rng_seed(42)
torch.manual_seed(42)


class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, hidden_dim):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.fc1 = nn.Linear(x_dim + c_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x, c):
        x = x.reshape(-1, self.x_dim)
        c = c.reshape(-1, self.c_dim)
        xc = torch.cat([x, c], dim=-1)

        hidden = self.softplus(self.fc1(xc))
        z_loc = self.fc21(hidden)
        # z_scale = torch.exp(self.fc22(hidden))
        z_scale = F.softplus(self.fc22(hidden)) + 1e-4
        return z_loc, z_scale


class EncoderNoCond(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x, _):
        x = x.reshape(-1, self.x_dim)

        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = F.softplus(self.fc22(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, c_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + c_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x_dim))

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        hidden = self.softplus(self.fc1(zc))
        x_loc = self.fc21(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class DecoderVolScaling(nn.Module):
    def __init__(self, x_dim, z_dim, c_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + c_dim, hidden_dim)
        self.softplus = nn.Softplus()
        self.film_scale = nn.Linear(c_dim, hidden_dim)
        self.film_shift = nn.Linear(c_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, x_dim)

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        hidden = self.fc1(zc)
        scale = self.film_scale(c)
        shift = self.film_shift(c)
        hidden = self.softplus(hidden * scale + shift)
        x_loc = self.fc21(hidden)
        x_log_scale = self.fc22(hidden)
        return x_loc, x_log_scale


class LatentScale(nn.Module):
    def __init__(self, c_dim, z_dim, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, c):
        return (F.sigmoid(self.net(c)) * 20) + 1e-4


class CVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.encoder = Encoder(x_dim, c_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, c_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, c, annealing_factor=1):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, log_x_scale = self.decoder.forward(z, c)
            x_scale = torch.exp(log_x_scale).expand_as(x_loc)

            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )
            return x_loc

    def guide(self, x, c, annealing_factor=1):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x, c)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct(self, x, c):
        z_loc, z_scale = self.encoder(x, c)
        z = dist.Normal(z_loc, z_scale).sample()
        x_loc, _ = self.decoder(z, c)
        return x_loc

    def reconstruct_map(self, x, c):
        z_loc, _ = self.encoder(x, c)
        x_loc, _ = self.decoder(z_loc, c)
        return x_loc

    def counterfactual_prediction(self, x, c1, c2):
        z_loc, _ = self.encoder(x, c1)
        x_loc, _ = self.decoder(z_loc, c2)
        return x_loc


class CVAEVolClustering(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.encoder = Encoder(x_dim, c_dim, z_dim, hidden_dim)
        self.decoder = DecoderVolScaling(x_dim, z_dim, c_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, c, annealing_factor=1):
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, log_x_scale = self.decoder.forward(z, c)
            x_scale = torch.exp(log_x_scale)

            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )
            return x_loc

    def guide(self, x, c, annealing_factor=1):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x, c)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct(self, x, c):
        z_loc, z_scale = self.encoder(x, c)
        z = dist.Normal(z_loc, z_scale).sample()
        x_loc, _ = self.decoder(z, c)
        return x_loc

    def reconstruct_map(self, x, c):
        z_loc, _ = self.encoder(x, c)
        x_loc, _ = self.decoder(z_loc, c)
        return x_loc

    def counterfactual_prediction(self, x, c1, c2):
        z_loc, _ = self.encoder(x, c1)
        x_loc, _ = self.decoder(z_loc, c2)
        return x_loc



class CVAEEteroschPrior(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.encoder = Encoder(x_dim, c_dim, z_dim, hidden_dim)
        # self.encoder = EncoderNoCond(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, c_dim, hidden_dim)
        self.prior_scale = LatentScale(c_dim, z_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, c, annealing_factor=1):
        pyro.module("decoder", self.decoder)
        pyro.module("prior_scale", self.prior_scale)

        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = self.prior_scale(c)

            with pyro.poutine.scale(scale=annealing_factor):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, log_x_scale = self.decoder.forward(z, c)
            x_scale = torch.exp(log_x_scale)

            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )
            return x_loc

    def guide(self, x, c, annealing_factor=1):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x, c)
            with pyro.poutine.scale(scale=annealing_factor):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct(self, x, c):
        z_loc, z_scale = self.encoder(x, c)
        z = dist.Normal(z_loc, z_scale).sample()
        x_loc, _ = self.decoder(z, c)
        return x_loc

    def reconstruct_map(self, x, c):
        z_loc, _ = self.encoder(x, c)
        x_loc, _ = self.decoder(z_loc, c)
        return x_loc

    def counterfactual_prediction(self, x, c1, c2):
        z_loc, _ = self.encoder(x, c1)
        x_loc, _ = self.decoder(z_loc, c2)
        return x_loc


def train(
    data_loaders: Iterable,
    x_dim: int,
    c_dim: int,
    hidden_dim: int = 50,
    z_dim=2,
    beta=1,
    annealing_start=1,
    num_epochs=30,
    heteroscedastic=False,
    dhsic_lambda=0.0,
    test_frequency=5,
    learning_rate=1.0e-3,
    cuda=False,
):
    # clear param store
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    # setup the VAE
    if heteroscedastic:
        # vae = CVAEVolClustering(x_dim=x_dim, c_dim=c_dim, use_cuda=cuda, z_dim=z_dim, hidden_dim=hidden_dim)
        vae = CVAEEteroschPrior(
            x_dim=x_dim, c_dim=c_dim, use_cuda=cuda, z_dim=z_dim, hidden_dim=hidden_dim
        )
    else:
        vae = CVAE(
            x_dim=x_dim, c_dim=c_dim, use_cuda=cuda, z_dim=z_dim, hidden_dim=hidden_dim
        )

    # setup the optimizer
    adam_args = {"lr": learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    if dhsic_lambda > 0:
        elbo = make_elbo_hsic(dhsic_lambda)
    else:
        elbo = Trace_ELBO()

    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = {}
    test_elbo = {}
    # training loop
    for epoch in range(num_epochs):
        progress = min((epoch + 1) / num_epochs, 1.0)
        annealing_factor = (1 - progress) * annealing_start + progress * beta

        # initialize loss accumulator
        epoch_loss = 0.0
        for batch in train_loader:
            x, c = batch
            if cuda:
                x = x.cuda()
                c = c.cuda()
            epoch_loss += svi.step(x, c, annealing_factor)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % test_frequency == 0:
            test_loss = 0.0
            for batch in test_loader:
                x, c = batch
                if cuda:
                    x = x.cuda()
                    c = c.cuda()
                test_loss += svi.evaluate_loss(x, c, annealing_factor)

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
            plot_llk(train_elbo, test_elbo)

    return vae


if __name__ == "__main__":
    pass
