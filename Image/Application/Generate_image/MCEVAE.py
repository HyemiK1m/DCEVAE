import torch
from torch import nn
from block import Reshape, Flatten, Conv_block
import torch.distributions as dists
import torch.nn.functional as F
import numpy as np

class MCEVAE(nn.Module):
    def __init__(self, args, sens_dim, rest_dim, des_dim, u_dim, KOF=32, p=0.04, batch_size=64):
        super(MCEVAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.int = args.int

        self.sens_dim = sens_dim
        self.rest_dim = rest_dim
        self.des_dim = des_dim

        self.u_dim = u_dim

        self.batch_size = batch_size

        """Encoder Network"""
        # xa to ur
        # xa to ur
        KOF = 32
        ka_size = int(KOF / 2)
        kx_size = KOF - ka_size

        self.encoder_x_to_kx = nn.Sequential()
        self.encoder_x_to_kx.add_module("block01", Conv_block(kx_size, 3, kx_size, 4, 2, 1, p=p))
        # self.encoder_x_to_kx.add_module("block02", Conv_block(kx_size, kx_size, kx_size, 4, 2, 1, p=p))

        self.encoder_a_to_ka = nn.Sequential()
        # Bx2 -> Bx2x1x1
        self.encoder_a_to_ka.add_module("reshape", Reshape((-1, 2, 1, 1)))
        # Bx2x1x1 -> Bxkax4x4
        self.encoder_a_to_ka.add_module("block00", Conv_block(ka_size, 2, ka_size, 32, 1, 0, p=p, transpose=True))

        self.encoder = nn.Sequential()
        # self.encoder.add_module("block01", Conv_block(KOF, 3, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(KOF * 2, KOF, KOF * 2, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(KOF * 2, KOF * 2, KOF * 2, 4, 2, 1, p=p))
        self.encoder.add_module("flatten", Flatten())
        self.encoder.add_module("FC01", nn.Linear(KOF * 32, 256))
        self.encoder.add_module("ReLU", nn.ReLU())
        self.encoder.add_module("FC02", nn.Linear(256, 2 * u_dim))

        """Decoder Network"""
        ku_size = KOF  # int(KOF * 2 * (u2_dim)/u1_dim)
        ka_size = 2 * KOF - ku_size  # 2 * KOF - ku_size

        self.decoder_a_to_ka = nn.Sequential()
        # Bx2 -> Bx2x1x1
        self.decoder_a_to_ka.add_module("reshape", Reshape((-1, 2, 1, 1)))
        # Bx2x1x1 -> Bxkax4x4
        self.decoder_a_to_ka.add_module("block00", Conv_block(ka_size, 2, ka_size, 4, 1, 0, p=p, transpose=True))

        self.decoder_u_to_ku = nn.Sequential()
        self.decoder_u_to_ku.add_module("block00", nn.Sequential(
            nn.Linear(u_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        ))  # Bx(ur+ud) -> Bxku
        self.decoder_u_to_ku.add_module("reshape", Reshape((-1, 256, 1, 1)))  # Bxkux1X1
        self.decoder_u_to_ku.add_module("block01", Conv_block(ku_size, 256, ku_size, 4, 1, 0, p=p, transpose=True))
        self.decoder_u_to_x = nn.Sequential()
        self.decoder_u_to_x.add_module("block01", Conv_block(KOF * 2, KOF * 2, KOF * 2, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block02", Conv_block(KOF, KOF * 2, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block03", Conv_block(KOF, KOF, KOF, 4, 2, 1, p=p, transpose=True))
        self.decoder_u_to_x.add_module("block04", Conv_block(3, KOF, 3, 4, 2, 1, p=p, transpose=True))

        """Classifier"""
        self.decoder_u_to_rest = nn.Sequential(
            nn.Linear(u_dim, u_dim),
            nn.ReLU(),
            nn.Linear(u_dim, rest_dim)
        )

        self.decoder_u_to_des = nn.Sequential(
            nn.Linear(u_dim, u_dim),
            nn.ReLU(),
            nn.Linear(u_dim, des_dim)
        )

        self.decoder_u_to_a = nn.Sequential(
            nn.Linear(u_dim, u_dim),
            nn.ReLU(),
            nn.Linear(u_dim, sens_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def D(self, z):
        return self.discriminator(z).squeeze()

    def q_u(self, x, a, r, d):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """
        intervention = torch.cat([1 - a, a], 1)
        ka = self.encoder_a_to_ka(intervention)
        kx = self.encoder_x_to_kx(x)
        xa = torch.cat([kx, ka], 1)
        stats = self.encoder(xa)
        u_mu = stats[:, :self.u_dim]
        u_logvar = stats[:, self.u_dim:]
        return u_mu, u_logvar

    def p_x(self, a, u):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parametersc of p(x|z)    (MB, inp_dim)
        """
        intervention = torch.cat([1 - a, a], 1)
        ku = self.decoder_u_to_ku(u)
        ka = self.decoder_a_to_ka(intervention)
        u = torch.cat([ku, ka], 1)
        x_hat = self.decoder_u_to_x(u)

        intervention_cf = 1 - intervention
        ka_cf = self.decoder_a_to_ka(intervention_cf)
        u = torch.cat([ku, ka_cf], 1)
        x_cf_hat = self.decoder_u_to_x(u)
        return x_hat, x_cf_hat

    def classifier(self, u):
        """classifier"""

        rest_ur = self.decoder_u_to_rest(u)
        a_pred = self.decoder_u_to_a(u)

        if self.int == 'M':
            return rest_ur, a_pred  # des_ud
        elif self.int == 'S':
            des_ud = self.decoder_u_to_des(u)
            return rest_ur, a_pred, des_ud

    def forward(self, x, a, r, d):
        """
        Encode the image, sample z and decode
        :param x: input image
        :return: parameters of p(x|z_hat), z_hat, parameters of q(z|x)
        """
        u_mu, u_logvar = self.q_u(x, a, r, d)
        u = self.reparameterize(u_mu, u_logvar)
        x_hat, x_cf_hat = self.p_x(a, u)
        # x_rec = self.reparameterize(x_mu, x_logvar)
        # x_cf = self.reparameterize(x_mu_cf, x_logvar_cf)

        return x_hat, x_cf_hat, u_mu, u_logvar, u

    def sampling_intervention(self, a):
        num = a.shape[0]
        u_mu = torch.zeros(num, self.u_dim).to(self.device)
        u_logvar = torch.ones(num, self.u_dim).to(self.device)
        u = self.reparameterize(u_mu, u_logvar)

        x_hat, _ = self.p_x(a, u)
        x_hat = nn.Sigmoid()(x_hat)
        return x_hat

    def sampling_counterfactual(self, x, a, r, d):
        u_mu, u_logvar = self.q_u(x, a, r, d)
        u = self.reparameterize(u_mu, u_logvar)

        x_hat, x_cf_hat = self.p_x(a, u)
        x_cf_hat = nn.Sigmoid()(x_cf_hat)
        return x_cf_hat

    def compute_kernel(self, x, y):
        # x: [4,8]
        # y: [4,8]
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # [4,1,8]
        y = y.unsqueeze(0) # [1,4,8]
        tiled_x = x.expand(x_size, y_size, dim) # [4, 4,8]
        tiled_y = y.expand(x_size, y_size, dim) # [4, 4, 8]
        # https://github.com/ShengjiaZhao/InfoVAE/blob/58f41c202049ceb2dbbd58336f92adc829d13200/elbo_bound.py
        # kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim) # .mean(2)? and /float(dim)? (recon loss: sum)
        kernel_input = (tiled_x - tiled_y).pow(2).sum(2)
        # print(kernel_input.size()) # [4,4]
        return torch.exp(-kernel_input)

    def compute_mmd(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        denominator = np.maximum(x_size*(x_size-1)/2,1) # if x_size=1, denominator get zero and it occurs error
        # denominator = torch.FloatTensor(x_size*(x_size-1)/2).to(self.device)

        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.sum() + y_kernel.sum() - 2 * xy_kernel.sum()
        mmd = mmd/denominator
        # print(mmd,denominator)
        # mmd = x_kernel.sum()/denominator + y_kernel.sum()/denominator - 2 * xy_kernel.sum()/denominator
        # mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def reconstruct_x(self, x, a, r, d):
        x_hat, _, _, _, _ = self.forward(x, a, r, d)
        x_hat = nn.Sigmoid()(x_hat)
        return x_hat

    def diagonal(self, M):
        """
        If diagonal value is close to 0, it could makes cholesky decomposition error.
        To prevent this, I add some diagonal value which won't affect the original value too much.
        """
        new_M = torch.where(torch.isnan(M), (torch.ones_like(M) * 1e-05).to(self.device), M)
        new_M = torch.where(torch.abs(new_M) < 1e-05, (torch.ones_like(M) * 1e-05).to(self.device), new_M)

        return new_M

    def image(self, x, sens, rest, des):
        x_fc, x_cf, u_mu, u_logvar, u = self.forward(x, sens, rest, des)
        #         x_fc = self.reparameterize(x_mu, x_logvar)
        #         x_cf = self.reparameterize(x_mu_cf, x_logvar_cf)
        x_fc = nn.Sigmoid()(x_fc)
        x_cf = nn.Sigmoid()(x_cf)
        return x_fc, x_cf

    def calculate_loss(self, x, sens, rest, des=None, beta1=20, beta2=1, beta3=0.2, beta4=1):
        """
        Given the input batch, compute the negative ELBO
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: -RE + beta * KL  (MB, ) or (1, )
        """

        # divide a
        MB = x.shape[0]

        x_p, x_cf_p, u_mu, u_logvar, u = self.forward(x, sens, rest, des)

        #         x_logvar = self.diagonal(x_logvar)
        u_logvar = self.diagonal(u_logvar)

        #         assert (torch.sum(torch.isnan(x_logvar)) == 0), 'x_logvar'
        assert (torch.sum(torch.isnan(u_logvar)) == 0), 'u_logvar'

        assert (torch.sum(torch.isnan(x_p)) == 0), 'x_p'
        assert (torch.sum(torch.isnan(x_cf_p)) == 0), 'x_cf_p'

        """x_reconstruction_loss"""
        x_recon = nn.BCEWithLogitsLoss(reduction='sum')(x_p, x) / MB
        x_cf_recon = nn.BCEWithLogitsLoss(reduction='sum')(x_cf_p, x) / MB
        assert (torch.sum(torch.isnan(x_recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(x_cf_recon)) == 0), 'x_cf_recon'

        """mmd loss"""
        a0_index = (sens == 0).nonzero()[:, 0].to(self.device)
        a1_index = (sens == 1).nonzero()[:, 0].to(self.device)
        u0 = u[a0_index, :]
        u1 = u[a1_index, :]

        u_sample = torch.rand(u_mu.shape).to(self.device)
        u0_sample = torch.rand(u0.shape).to(self.device)
        u1_sample = torch.rand(u1.shape).to(self.device)

        if u0.shape[0] == 0 or u1.shape[0] == 0:
            mmd = torch.zeros_like(x_recon)
        else:
            mmd = self.compute_mmd(u_sample, u)

        mmd_A0 = self.compute_mmd(u0_sample, u0) * u0.shape[0] if u0.shape[0] != 0 else torch.zeros_like(x_recon)
        mmd_A1 = self.compute_mmd(u1_sample, u1) * u1.shape[0] if u1.shape[0] != 0 else torch.zeros_like(x_recon)
        mmd_a = mmd_A0 + mmd_A1

        """Classifier loss"""
        if self.int == 'M':
            rest_ur, a_pred = self.classifier(u)
            recon_rest_ur = nn.BCEWithLogitsLoss(reduction='sum')(rest_ur, rest) / MB
            recon_sens = nn.BCEWithLogitsLoss(reduction='sum')(a_pred, sens) / MB
            l_recon = recon_rest_ur + recon_sens
        elif self.int == 'S':
            rest_ur, a_pred, des_ud = self.classifier(u)
            recon_rest_ur = nn.BCEWithLogitsLoss(reduction='sum')(rest_ur, rest) / MB
            recon_sens = nn.BCEWithLogitsLoss(reduction='sum')(a_pred, sens) / MB
            recon_des_ud = nn.BCEWithLogitsLoss(reduction='sum')(des_ud, des) / MB
            l_recon = recon_rest_ur + recon_sens + recon_des_ud

        ELBO = beta1 * (x_recon) + beta2 * l_recon + beta3 * mmd + beta4 * mmd_a

        return ELBO, x_recon, l_recon, mmd, mmd_a

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mu)