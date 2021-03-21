from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, SplitGConv2D
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
import torch
import torch.nn as nn
import torch.nn.functional as F


class AllConvNet(nn.Module):
    """
    Model architecture:
        Large All-CNN for CIFAR-10 (https://arxiv.org/abs/1412.6806)
        input:      3 channel input image 96x96
        conv1:      2x2 conv. 320 LeakyReLU, stride 1
        conv2:      2x2 conv. 320 LeakyReLU, stride 1
        conv3:      2x2 conv. 320 LeakyReLU, stride 2
        conv4:      2x2 conv. 640 LeakyReLU, stride 1, dropout 0.1
        conv5:      2x2 conv. 640 LeakyReLU, stride 1, dropout 0.1
        conv6:      2x2 conv. 640 LeakyReLU, stride 2
        conv7:      2x2 conv. 960 LeakyReLU, stride 1, dropout 0.2
        conv8:      2x2 conv. 960 LeakyReLU, stride 1, dropout 0.2
        conv9:      2x2 conv. 960 LeakyReLU, stride 2
        conv10:     2x2 conv. 1280 LeakyReLU, stride 1, dropout 0.3
        conv11:     2x2 conv. 1280 LeakyReLU, stride 1, dropout 0.3
        conv12:     2x2 conv. 1280 LeakyReLU, stride 2
        conv13:     2×2 conv. 1600 LeakyReLU, stride 1, dropout 0.4
        conv14:     2×2 conv. 1600 LeakyReLU, stride 1, dropout 0.4
        conv15:     2×2 conv. 1600 LeakyReLU, stride 2
        conv16:     2×2 conv. 1920 LeakyReLU, stride 1, dropout 0.5
        conv17:     1×1 conv. 1920 LeakyReLU, stride 1, dropout 0.5
        softmax:    2-way softmax / sigmoid
    """

    def __init__(self, input_channels: int = 1, no_classes: int = 10) -> None:
        super(AllConvNet, self).__init__()

        self.in_channels = input_channels
        self.classes = no_classes

        self.conv1 = nn.Conv2d(self.in_channels, 320, 2, stride=1)
        self.conv2 = nn.Conv2d(320, 320, 2, stride=1)
        # increase stride as replacement for pooling
        self.conv3 = nn.Conv2d(320, 320, 2, stride=2)
        self.conv4 = nn.Conv2d(320, 640, 2, stride=1)
        self.conv5 = nn.Conv2d(640, 640, 2, stride=1)
        # increase stride as replacement for pooling
        self.conv6 = nn.Conv2d(640, 640, 2, stride=2)
        self.conv7 = nn.Conv2d(640, 960, 2, stride=1)
        self.conv8 = nn.Conv2d(960, 960, 2, stride=1)
        # increase stride as replacement for pooling
        self.conv9 = nn.Conv2d(960, 960, 2, stride=2)
        self.conv10 = nn.Conv2d(960, 1280, 2, stride=1)
        self.conv11 = nn.Conv2d(1280, 1280, 2, stride=1)
        # increase stride as replacement for pooling
        self.conv12 = nn.Conv2d(1280, 1280, 2, stride=2)
        self.conv13 = nn.Conv2d(1280, 1600, 2, stride=1)
        self.conv14 = nn.Conv2d(1600, 1600, 2, stride=1)
        # increase stride as replacement for pooling
        self.conv15 = nn.Conv2d(1600, 1600, 2, stride=2)
        self.conv16 = nn.Conv2d(1600, 1920, 1, stride=1)
        self.class_conv = nn.Conv2d(1920, self.classes, 1, stride=1)

        # initialize weight and bias
        for m in self.modules():
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.dropout(x, p=0.1)
        x = F.leaky_relu(self.conv4(x))
        x = F.dropout(x, p=0.1)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.conv7(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        x = F.dropout(x, p=0.3)
        x = F.leaky_relu(self.conv10(x))
        x = F.dropout(x, p=0.3)
        x = F.leaky_relu(self.conv11(x))
        x = F.leaky_relu(self.conv12(x))
        x = F.dropout(x, p=0.4)
        x = F.leaky_relu(self.conv13(x))
        x = F.dropout(x, p=0.4)
        x = F.leaky_relu(self.conv14(x))
        x = F.leaky_relu(self.conv15(x))
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.conv16(x))
        x = F.dropout(x, p=0.5)
        class_out = F.leaky_relu(self.class_conv(x))
        print(class_out.size())
        # simulate avg pooling
        pool_out = class_out.reshape(class_out.size(0),
                                     class_out.size(1), -1).mean(-1)
        return F.softmax(pool_out, dim=1)


class AllGConvNet(nn.Module):
    def __init__(self, input_channels: int = 1, no_classes: int = 10) -> None:
        super(AllGConvNet, self).__init__()

        self.in_channels = input_channels
        self.classes = no_classes

        # make_gconv_indices.py forces kernel size to be uneven
        #
        # (1) this model shows a huge demand on memory
        #     > reduce from 17 to the initial 9 layers of All Conv Net
        #
        # (2) training of an epoch takes 2.5 hours
        #     > reduce the number of layers to 6
        #
        self.gconv1 = P4ConvZ2(self.in_channels, 96, 3, stride=1)
        self.gconv2 = P4ConvP4(96, 96, 3, stride=2)
        self.gconv3 = P4ConvP4(96, 192, 3, stride=1)
        self.gconv4 = P4ConvP4(192, 192, 3, stride=2)
        self.gconv5 = P4ConvP4(192, 384, 1, stride=1)
        self.class_gconv = P4ConvP4(384, self.classes, 1, stride=1)

        # initialize weight and bias
        for m in self.modules():
            if isinstance(m, SplitGConv2D):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                # commented after manual inspection of one Tensor
                # nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.gconv1(x))
        x = F.leaky_relu(self.gconv2(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.gconv3(x))
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(self.gconv4(x))
        x = F.dropout(x, p=0.3)
        x = F.leaky_relu(self.gconv5(x))
        x = F.dropout(x, p=0.3)
        x = F.dropout(x, p=0.5)
        class_out = F.leaky_relu(self.class_gconv(x))
        pool_out = class_out.reshape(class_out.size(0),
                                     class_out.size(1), -1).mean(-1)
        return F.softmax(pool_out, dim=1)


class GConvNet(nn.Module):
    # example taken from
    # https://github.com/adambielski/pytorch-gconv-experiments/blob/master/mnist/mnist.py
    def __init__(self, input_channels: int = 1, no_classes: int = 10) -> None:
        super(GConvNet, self).__init__()

        self.in_channels = input_channels
        self.classes = no_classes

        # make_gconv_indices.py forces kernel size to be uneven
        # first tests on CIFAR10 used doubled size of inner layers
        # (starting with 96 output channels in gconv1)
        # PCam dataset was trained using half sized to reduce memory consumption
        self.gconv1 = P4ConvZ2(self.in_channels, 48, 3)
        self.gconv2 = P4ConvP4(48, 48, 3)
        self.gconv3 = P4ConvP4(48, 96, 3)
        self.gconv4 = P4ConvP4(96, 96, 3)
        self.gconv5 = P4ConvP4(96, 192, 3)
        self.gconv6 = P4ConvP4(192, 192, 3)
        # pooling reduced the image to 17x17 from gconv1 to gconv6
        self.fc1 = nn.Linear(17*17*4*192, 512)
        self.fc2 = nn.Linear(512, self.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.gconv3(x))
        x = F.relu(self.gconv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.dropout(x, p=0.3)
        x = F.relu(self.gconv5(x))
        x = F.relu(self.gconv6(x))
        # reshape
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4)
        x = self.fc2(x)
        return x
