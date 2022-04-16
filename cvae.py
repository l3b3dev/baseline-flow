import torch
import torch.nn.functional as F


##########################
# MODEL
##########################


def to_onehot(labels, num_classes, device):
    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot


class ConditionalVariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_features, num_hidden_1, num_latent, num_classes):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.num_classes = num_classes

        # ENCODER
        self.hidden_1 = torch.nn.Linear(num_features + num_classes, num_hidden_1)
        self.z_mean = torch.nn.Linear(num_hidden_1, num_latent)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = torch.nn.Linear(num_hidden_1, num_latent)

        # DECODER
        self.linear_3 = torch.nn.Linear(num_latent + num_classes, num_hidden_1)
        self.linear_4 = torch.nn.Linear(num_hidden_1, num_features + num_classes)

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def encoder(self, features, targets):
        # Add condition
        onehot_targets = to_onehot(targets, self.num_classes, self.device)
        x = torch.cat((features, onehot_targets), dim=1)

        # ENCODER
        x = self.hidden_1(x)
        x = F.leaky_relu(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def decoder(self, encoded, targets):
        # Add condition
        onehot_targets = to_onehot(targets, self.num_classes, self.device)
        encoded = torch.cat((encoded, onehot_targets), dim=1)

        # DECODER
        x = self.linear_3(encoded)
        x = F.leaky_relu(x)
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, features, targets):
        z_mean, z_log_var, encoded = self.encoder(features, targets)
        decoded = self.decoder(encoded, targets)

        return z_mean, z_log_var, encoded, decoded
