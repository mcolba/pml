# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

from src.vae.utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

pyro.set_rng_seed(42)
torch.manual_seed(42)


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, self.x_dim)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        # z_scale = torch.exp(self.fc22(hidden))
        z_scale = F.softplus(self.fc22(hidden)) + 1e-4
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()
        self.log_x_scale = nn.Parameter(torch.zeros(x_dim))

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        x_loc = self.fc21(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class VAE(nn.Module):
    def __init__(
        self, x_dim, z_dim=50, hidden_dim=400, use_cuda=False, prior_t_df=np.inf
    ):
        super().__init__()
        self.x_dim = x_dim
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.prior_t_df = prior_t_df

    # define the model p(x|z)p(z)
    def model(self, x, anneling_factor=1):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)

            with pyro.poutine.scale(scale=anneling_factor):
                if np.isinf(self.prior_t_df):
                    z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
                else:
                    z = pyro.sample(
                        "latent",
                        dist.StudentT(self.prior_t_df, z_loc, z_scale).to_event(1),
                    )

            # decode the latent code z
            x_loc, log_x_scale = self.decoder.forward(z)
            x_scale = torch.exp(log_x_scale)

            # score against actual data
            pyro.sample(
                "obs",
                dist.Normal(x_loc, x_scale, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )
            # return the loc so we can visualize it later
            return x_loc

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, anneling_factor=1):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            with pyro.poutine.scale(scale=anneling_factor):
                if np.isinf(self.prior_t_df):
                    pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
                else:
                    pyro.sample(
                        "latent",
                        dist.StudentT(self.prior_t_df, z_loc, z_scale).to_event(1),
                    )

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        x_loc, _ = self.decoder(z)
        return x_loc

        # define a helper function for reconstructing images

    def reconstruct_map(self, x):
        z_loc, _ = self.encoder(x)
        x_loc, _ = self.decoder(z_loc)
        return x_loc


def train(
    data_loaders: Iterable,
    x_dim: int,
    hidden_dim: int = 50,
    z_dim=2,
    beta=1,
    anneling_start=1,
    num_epochs=30,
    test_frequency=5,
    learning_rate=1.0e-3,
    cuda=False,
    jit=False,
    visdom_flag=False,
    tsne_iter=100,
    prior_t_df=np.inf,
):
    # clear param store
    pyro.clear_param_store()

    train_loader, test_loader = data_loaders

    # setup the VAE
    vae = VAE(
        x_dim=x_dim,
        use_cuda=cuda,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        prior_t_df=prior_t_df,
    )

    # setup the optimizer
    adam_args = {"lr": learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    # setup visdom for visualization
    if visdom_flag:
        vis = visdom.Visdom()

    train_elbo = {}
    test_elbo = {}
    # training loop
    for epoch in range(num_epochs):
        #
        progress = min((epoch + 1) / num_epochs, 1.0)
        anneling_factor = (1 - progress) * anneling_start + progress * beta

        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for (x,) in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x, anneling_factor)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo[epoch] = total_epoch_loss_train
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.0
            # compute the loss over the entire test set
            for i, (x,) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x, anneling_factor)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if visdom_flag:
                        plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(
                                test_img.reshape(28, 28).detach().cpu().numpy(),
                                opts={"caption": "test image"},
                            )
                            vis.image(
                                reco_img.reshape(28, 28).detach().cpu().numpy(),
                                opts={"caption": "reconstructed image"},
                            )

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo[epoch] = total_epoch_loss_test
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
            plot_llk(train_elbo, test_elbo)

        if epoch == tsne_iter:
            mnist_test_tsne(vae=vae, test_loader=test_loader)

    return vae


if __name__ == "__main__":
    pass
