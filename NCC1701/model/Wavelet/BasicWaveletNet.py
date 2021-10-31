import torch
import torch.nn as nn
import torch.nn.functional as tf
import torchvision.models as models

from Math.TorchWaveletTransforms import DWTForward, DWTInverse

class BasicResBlock(nn.Module):
    def __init__(self, in_channel=1, do_drop=True):
        super(BasicResBlock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.tanh1= nn.PReLU(32)
        
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.tanh2 = nn.PReLU(64)
        
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.tanh3 = nn.PReLU(128)
          
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.tanh4 = nn.PReLU(256)

        self.cnn5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.tanh5 = nn.PReLU(128)

        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.tanh6 = nn.PReLU(64)

        self.cnn7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.tanh7 = nn.PReLU(32)

        self.cnnout = nn.Conv2d(in_channels=32, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        self.tanhout = nn.PReLU(1)

        self.do_drop = do_drop
        self.drop = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x):

        out = self.cnn1(x)
        out = self.tanh1(out)

        out = self.cnn2(out)
        out = self.tanh2(out)

        if self.do_drop:
            out = self.drop(out)

        out = self.cnn3(out)
        out = self.tanh3(out)

        out = self.cnn4(out)
        out = self.tanh4(out)

        if self.do_drop:
            out = self.drop(out)

        out = self.cnn5(out)
        out = self.tanh5(out)

        out = self.cnn6(out)
        out = self.tanh6(out)

        if self.do_drop:
            out = self.drop(out)

        out = self.cnn7(out)
        out = self.tanh7(out)

        out = self.cnnout(out)
        out = self.tanhout(out)

        return out+x
 
class MiniResNet(nn.Module):
    def __init__(self, do_drop=False, n_resblocks=2, n_channels=1):
        super(MiniResNet, self).__init__()
        
        model = []
        for _ in range(n_resblocks):
            model += [BasicResBlock(n_channels, do_drop)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Image2ImageNet(nn.Module):
    """description of class"""

    def __init__(self, n_channels=1, trainable_wavelets=False, do_drop=False):
        super(Image2ImageNet, self).__init__()

        self.ApproxCoeffsNet = MiniResNet(do_drop, n_channels=1)
        self.DetailsCoeffsNet = MiniResNet(do_drop, n_channels=3)

        self.dwt = DWTForward(trainable=trainable_wavelets)
        self.idwt = DWTInverse(trainable=trainable_wavelets)
        
    def forward(self, images):
        ApproxCoeffs, DetailsCoeffs = self.dwt(images)            
        
        DetailsCoeffs = torch.squeeze(DetailsCoeffs, 0)
        ApproxCoeffsOut = self.ApproxCoeffsNet(ApproxCoeffs)   
        DetailsCoeffsOut = self.DetailsCoeffsNet(DetailsCoeffs)
        DetailsCoeffsOut = DetailsCoeffsOut.unsqueeze(0)

        out = self.idwt((ApproxCoeffsOut,[DetailsCoeffsOut]))

        return out

class TorchvisionClassifier(nn.Module):
    def __init__(self, n_channels, n_class, base_model=models.resnet18, is_pretrained=True):
        super(TorchvisionClassifier, self).__init__()
        
        base_model = base_model(pretrained=is_pretrained)
        if n_channels != 3:
            base_model.conv1 = nn.Conv2d(n_channels, out_channels=base_model.conv1.out_channels, kernel_size=base_model.conv1.kernel_size,
                                    stride=base_model.conv1.stride, padding=base_model.conv1.padding, bias=base_model.conv1.bias)

        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=n_class, bias=True)

        self.model = base_model

    def forward(self, input):
        return self.model(input)

class Image2ClassifyNet(nn.Module):
    """description of class"""

    def __init__(self, n_channels=1, n_class=10, trainable_wavelets=False, do_drop=False, classiferNet=TorchvisionClassifier):
        super(Image2ClassifyNet, self).__init__()

        self.ApproxCoeffsNet = MiniResNet(do_drop, n_channels=1)
        self.DetailsCoeffsNet = MiniResNet(do_drop, n_channels=3)
        self.CombiedCoeffsNet = MiniResNet(do_drop, n_channels=4)

        self.dwt = DWTForward(trainable=trainable_wavelets)

        self.classifier = classiferNet(n_channels=4, n_class=n_class)
        
    def forward(self, images):
        ApproxCoeffs, DetailsCoeffs = self.dwt(images)            
        
        DetailsCoeffs = torch.squeeze(DetailsCoeffs, 1)
        ApproxCoeffsOut = self.ApproxCoeffsNet(ApproxCoeffs)   
        DetailsCoeffsOut = self.DetailsCoeffsNet(DetailsCoeffs)
        CombinedCoeffsOut = torch.cat([ApproxCoeffsOut,DetailsCoeffsOut], dim=1)
        WaveOut = self.CombiedCoeffsNet(CombinedCoeffsOut)

        return tf.relu_(self.classifier(WaveOut))

class ImageCorrectNClassifyNet(nn.Module):
    """description of class"""

    def __init__(self, n_channels=1, n_class=10, trainable_wavelets=False, do_drop=False, classiferNet=TorchvisionClassifier, branch_split=False, relu_at_end=True):
        super(ImageCorrectNClassifyNet, self).__init__()

        self.ApproxCoeffsNet = MiniResNet(do_drop, n_channels=1)
        self.DetailsCoeffsNet = MiniResNet(do_drop, n_channels=3)
        self.CombiedCoeffsNet = MiniResNet(do_drop, n_channels=4)

        self.dwt = DWTForward(trainable=trainable_wavelets)
        self.idwt = DWTInverse(trainable=trainable_wavelets)

        self.branch_split = branch_split
        if branch_split:
            self.classifier = classiferNet(n_channels=4, n_class=n_class)
        else:
            self.classifier = classiferNet(n_channels=8, n_class=n_class)

        self.relu_at_end = relu_at_end
        
    def forward(self, images):
        ApproxCoeffs, DetailsCoeffs = self.dwt(images)     
        
        DetailsCoeffs = torch.squeeze(DetailsCoeffs, 1)
        ApproxCoeffsOut = self.ApproxCoeffsNet(ApproxCoeffs)   
        DetailsCoeffsOut = self.DetailsCoeffsNet(DetailsCoeffs)
        CombinedCoeffsOut = torch.cat([ApproxCoeffsOut,DetailsCoeffsOut], dim=1)
        WaveOut = self.CombiedCoeffsNet(CombinedCoeffsOut)

        WaveIn = torch.cat([ApproxCoeffs,DetailsCoeffs], dim=1)
        if self.branch_split:
            Wave4Classify = WaveOut
        else:
            Wave4Classify = torch.cat([WaveOut,WaveIn], dim=1)

        ImgOut = self.idwt((ApproxCoeffsOut,[DetailsCoeffsOut.unsqueeze(1)]))
        ClassOut = self.classifier(Wave4Classify)
        if self.relu_at_end:
            ClassOut = tf.relu_(ClassOut)
        else:
            ClassOut = tf.sigmoid(ClassOut)


        return ClassOut, ImgOut