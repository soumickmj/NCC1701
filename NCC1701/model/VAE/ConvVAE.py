import os
import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, n_channels=1, batch_size=16, no_of_sample = 1, starting_n_features=16, ZDIMS=20, input_size=(128,128), fc_features=1024, up_while_trans=False):
        super(ConvVAE, self).__init__()

        self.in_fc_features = (starting_n_features*2) * input_size[0] * input_size[1]
        mid_fc_features = int(self.in_fc_features // math.sqrt((self.in_fc_features / fc_features)))

        self.starting_n_features = starting_n_features
        self.input_size = input_size
        self.batch_size = batch_size
        self.no_of_sample = no_of_sample

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=starting_n_features, kernel_size=3, padding=1, stride=1)  
        self.conv2 = nn.Conv2d(in_channels=starting_n_features, out_channels=starting_n_features*2, kernel_size=3, padding=1, stride=1)
        self.fc11 = nn.Linear(in_features=self.in_fc_features, out_features=mid_fc_features)
        self.fc12 = nn.Linear(in_features=mid_fc_features, out_features=fc_features)
        self.fc13 = nn.Linear(in_features=fc_features, out_features=ZDIMS)

        self.fc21 = nn.Linear(in_features=self.in_fc_features, out_features=mid_fc_features)
        self.fc22 = nn.Linear(in_features=mid_fc_features, out_features=fc_features)
        self.fc23 = nn.Linear(in_features=fc_features, out_features=ZDIMS)
        self.relu = nn.ReLU()

        # For decoder

        # For mu
        self.fc1 = nn.Linear(in_features=ZDIMS, out_features=fc_features)
        self.fc2 = nn.Linear(in_features=fc_features, out_features=mid_fc_features)
        if not up_while_trans:
            self.fc3 = nn.Linear(in_features=mid_fc_features, out_features=(input_size[0]//4) * (input_size[1]//4) * (starting_n_features*2))
            self.decoded_size = ((starting_n_features*2), (input_size[0]//4) , (input_size[1]//4))
            self.conv_t11 = nn.ConvTranspose2d(in_channels=starting_n_features*2, out_channels=starting_n_features, kernel_size=4, padding=1, stride=2)
            self.conv_t12 = nn.ConvTranspose2d(in_channels=starting_n_features, out_channels=n_channels, kernel_size=4, padding=1, stride=2)

            self.conv_t21 = nn.ConvTranspose2d(in_channels=starting_n_features*2, out_channels=starting_n_features, kernel_size=4, padding=1, stride=2)
            self.conv_t22 = nn.ConvTranspose2d(in_channels=starting_n_features, out_channels=n_channels, kernel_size=4, padding=1, stride=2)
        else:
            #Option 2 for mu:
            self.fc3 = nn.Linear(in_features=mid_fc_features, out_features=self.in_fc_features)
            self.decoded_size = (starting_n_features*2, ) + input_size
            self.conv_t11 = nn.ConvTranspose2d(in_channels=starting_n_features*2, out_channels=starting_n_features, kernel_size=3, padding=1, stride=1)
            self.conv_t12 = nn.ConvTranspose2d(in_channels=starting_n_features, out_channels=n_channels, kernel_size=3, padding=1, stride=1)

            self.conv_t21 = nn.ConvTranspose2d(in_channels=starting_n_features*2, out_channels=starting_n_features, kernel_size=3, padding=1, stride=1)
            self.conv_t22 = nn.ConvTranspose2d(in_channels=starting_n_features, out_channels=n_channels, kernel_size=3, padding=1, stride=1)

        #Parameter initialization
        for m in self.modules():
        
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
                init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #init.kaiming_uniform(m.weight.data)
                init.constant(m.bias, .1)
        
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def encode(self, x: Variable) -> (Variable, Variable):

        x = x.view(-1, 1, self.input_size[0] , self.input_size[1])
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, self.in_fc_features)

        mu_z = F.elu(self.fc11(x))
        mu_z = F.elu(self.fc12(mu_z))
        mu_z = self.fc13(mu_z)

        logvar_z = F.elu(self.fc21(x))
        logvar_z = F.elu(self.fc22(logvar_z))
        logvar_z = self.fc23(logvar_z)

        return mu_z, logvar_z

    def reparameterize(self, mu: Variable, logvar: Variable) -> list:
        """THE REPARAMETERIZATION IDEA:
        For each training sample (we get 128 batched at a time)
        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians
        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix
        Returns
        -------
        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.
        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(self.no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                # - std.data is the [128,ZDIMS] tensor that is wrapped by std
                # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
                #   and stddev 1 normal distribution that is 128 samples
                #   of random ZDIMS-float vectors
                eps = Variable(std.data.new(std.size()).normal_())
                # - sample from a normal distribution with standard
                #   deviation = std and mean = mu by multiplying mean 0
                #   stddev 1 sample with desired std and mu, see
                #   https://stats.stackexchange.com/a/16338
                # - so we have 128 sets (the batch) of random ZDIMS-float
                #   vectors sampled from normal distribution with learned
                #   std and mu for the current input
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> (Variable, Variable):

        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = x.view(-1, self.decoded_size[0], self.decoded_size[1], self.decoded_size[2])
        mu_x = F.relu(self.conv_t11(x))
        mu_x = F.sigmoid(self.conv_t12(mu_x))

        logvar_x = F.relu(self.conv_t11(x))
        logvar_x = F.sigmoid(self.conv_t12(logvar_x))

        return mu_x.view(-1, self.input_size[0]*self.input_size[1]), logvar_x.view(-1, self.input_size[0]*self.input_size[1])

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, self.input_size[0]*self.input_size[1]))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?

        if self.training:
            GLL = 0
            x = x.view(-1, self.input_size[0]*self.input_size[1])
            for recon_x_one in recon_x:
                mu_x, logvar_x = recon_x_one
                part1 = torch.sum(logvar_x) / self.batch_size
                sigma = logvar_x.mul(0.5).exp_()
                part2 = torch.sum(((x - mu_x) / sigma) ** 2) / self.batch_size
                GLL += .5 * (part1 + part2)

            GLL /= len(recon_x)
        else:
            x = x.view(-1, self.input_size[0]*self.input_size[1])
            mu_x, logvar_x = recon_x
            part1 = torch.sum(logvar_x) / self.batch_size
            sigma = logvar_x.mul(0.5).exp_()
            part2 = torch.sum(((x - mu_x) / sigma) ** 2) / self.batch_size
            GLL = .5 * (part1 + part2)



        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #exp is genareting infs
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar)
        # Normalise by same number of elements as in reconstruction
        KLD /= self.batch_size


        return GLL + KLD
