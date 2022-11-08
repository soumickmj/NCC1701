import math
import sys

import torch
import torch.nn as nn
import torchvision
# from utils.utils import *
from pytorch_msssim import MS_SSIM, SSIM

from .Resnet2D import ResNet
from .simpleunet import UNet
from .VesselSeg_UNet3d_DeepSup import U_Net_DeepSup


# currently configured for 1 channel only, with datarange as 1 for SSIM
class PerceptualLoss(torch.nn.Module):
    def __init__(self, device="cuda:0", loss_model="densenet161", n_level=math.inf, resize=None, loss_type="L1", mean=[], std=[]):
        super(PerceptualLoss, self).__init__()
        blocks = []

        if loss_model == "resnet2D":  # TODO: not finished
            model = ResNet(in_channels=1, out_channels=1).to(device)
            chk = torch.load(
                r"./Engineering/pLoss/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            model.load_state_dict(chk['state_dict'])
        elif loss_model == "unet2D":
            model = UNet(in_channels=1, out_channels=1, depth=5, wf=6, padding=True,
                         batch_norm=False, up_mode='upsample', droprate=0.0, is3D=False,
                         returnBlocks=False, downPath=True, upPath=True).to(device)
            chk = torch.load(
                r"./Engineering/pLoss/SimpleU_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.down_path[0].block.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[1].block.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[2].block.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[3].block.eval()
                    )
                )
        elif loss_model == "unet3Dds":
            model = U_Net_DeepSup().to(device)
            chk = torch.load(
                r"./Engineering/pLoss/VesselSeg_UNet3d_DeepSup.pth", map_location=device)
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.Conv1.conv.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool1.eval(),
                        model.Conv2.conv.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool2.eval(),
                        model.Conv3.conv.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool3.eval(),
                        model.Conv4.conv.eval()
                    )
                )
            if n_level >= 5:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool4.eval(),
                        model.Conv5.conv.eval()
                    )
                )
        elif loss_model == "resnext1012D":
            model = torchvision.models.resnext101_32x8d()
            model.conv1 = nn.Conv2d(
                1,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias is not None,
            )

            model.fc = nn.Linear(
                in_features=model.fc.in_features,
                out_features=33,
                bias=model.fc.bias is not None,
            )

            model.to(device)
            chk = torch.load(r"./Engineering/pLoss/ResNeXt-3-class-best-latest.pth", map_location=device)
            model.load_state_dict(chk)
            blocks.append(
                nn.Sequential(
                    model.conv1.eval(),
                    model.bn1.eval(),
                    model.relu.eval(),
                )
            )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.maxpool.eval(),
                        model.layer1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.layer2.eval())
            if n_level >= 4:
                blocks.append(model.layer3.eval())
            if n_level >= 5:
                blocks.append(model.layer4.eval())
        elif loss_model == "densenet161":
            sys.exit("Weights for DenseNet151 as PLN not available")
            model = torchvision.models.densenet161()
            model.features.conv0 = nn.Conv2d(
                1,
                model.features.conv0.out_channels,
                kernel_size=model.features.conv0.kernel_size,
                stride=model.features.conv0.stride,
                padding=model.features.conv0.padding,
                bias=model.features.conv0.bias is not None,
            )

            model.classifier = nn.Linear(
                in_features=model.classifier.in_features,
                out_features=33,
                bias=model.classifier.bias is not None,
            )

            model.to(device)
            # chk = torch.load(r"./Engineering/pLoss/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location=device)
            # model.load_state_dict(chk['state_dict'])
            model = model.features
            blocks.append(
                nn.Sequential(
                    model.conv0.eval(),
                    model.norm0.eval(),
                    model.relu0.eval(),
                )
            )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.pool0.eval(),
                        model.denseblock1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.denseblock2.eval())
            if n_level >= 4:
                blocks.append(model.denseblock3.eval())
            if n_level >= 5:
                blocks.append(model.denseblock4.eval())

        for bl in blocks:
            for params in bl.parameters():
                params.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        if (mean is not None and len(mean) > 1) and (std is not None and len(std) > 1) and (len(mean) == len(std)):
            self.mean = nn.Parameter(
                torch.tensor(mean).view(1, len(mean), 1, 1))
            self.std = nn.Parameter(torch.tensor(std).view(1, len(std), 1, 1))
        else:
            self.mean = None
            self.std = None
        self.resize = resize

        if loss_type == "L1":
            self.loss_func = torch.nn.functional.l1_loss
        elif loss_type == "MultiSSIM":
            self.loss_func = MS_SSIM(reduction='mean').to(device)
        elif loss_type == "SSIM3D":
            self.loss_func = SSIM(
                data_range=1, size_average=True, channel=1, spatial_dims=3).to(device)
        elif loss_type == "SSIM2D":
            self.loss_func = SSIM(
                data_range=1, size_average=True, channel=1, spatial_dims=2).to(device)

    def forward(self, input, target):
        if self.mean is not None:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='trilinear' if len(
                input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
            target = self.transform(target, mode='trilinear' if len(
                input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.loss_func(x, y)
        return loss


if __name__ == '__main__':
    x = PerceptualLoss(resize=None).cuda()
    a = torch.rand(2, 1, 24, 24).cuda()
    b = torch.rand(2, 1, 24, 24).cuda()
    l = x(a, b)
    sdsd
