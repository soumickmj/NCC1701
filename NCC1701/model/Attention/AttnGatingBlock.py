import torch
import torch.nn
import torch.nn.functional as F

__author__ = "Philipp Ernst, Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Philipp Ernst", "Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class AttnGatingBlock(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels, is_leaky=False):
        super(AttnGatingBlock, self).__init__()
        self.conv1 = nn.Conv2d(x_channels, inter_channels, 2, 2)
        self.conv2 = nn.Conv2d(g_channels, inter_channels, 1)
        self.conv3 = nn.Conv2d(inter_channels, 1, 1)
        self.conv4 = nn.Conv2d(x_channels, x_channels, 1)
        self.bn1 = nn.BatchNorm2d(x_channels)
        self.act_xg = nn.PReLU(inter_channels) if is_leaky else nn.ReLU()

    def forward(self, x, g):
        theta_x = self.conv1(x)
        phi_g = self.conv2(g)

        concat_xg = theta_x+phi_g
        act_xg = self.act_xg(concat_xg)
        psi = self.conv3(act_xg)
        sigmoid_xg = torch.sigmoid(psi)

        upsample_psi = F.interpolate(sigmoid_xg, scale_factor=2, mode='bilinear')
        upsample_psi = upsample_psi.repeat(1, x.shape[1], 1, 1)

        y = torch.mul(upsample_psi, x)
        result = self.conv4(y)
        result_bn = self.bn1(result)
        return result_bn