import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from pathlib import Path
import platform
import numpy as np


### all things pertaining to the resnet connecting input directly to output, skiping the UNet. ###
class ResNet_fixedImgSize(nn.Module):
    def __init__(self, bn=True, SingleChannel=False):
        super().__init__()
        ch_in = 1 if SingleChannel else 3
        self.res_layers = nn.Sequential(ResBlock(ch_in,  64, bn=bn),
                                        ResBlock(64, 64, bn=bn),
                                        ResBlock(64, 64, bn=bn),
                                        ResBlock(64, 64, bn=bn),
                                        )

    def forward(self, input):
        output: Tensor = self.res_layers(input)
        return output


def Conv2D_BN_ReLU(ch_in, ch_out, kernel=3, padding=1, stride=1, bn=True):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel, padding=padding, bias=False, stride=stride),
        BatchNorm2d_cond(ch_out, bn=bn),
        nn.ReLU(inplace=True),
    )


def BatchNorm2d_cond(ch_out, bn=True):
    if bn:
        return nn.BatchNorm2d(ch_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    else:
        return nn.Identity()


class ResBlock(nn.Module):
    # for different types of resnet blocks, see: https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec.
    def __init__(self, ch_in, ch_out, bn=True, expansion=False):
        super().__init__()
        self.ch_in, self.ch_out, self.expansion = ch_in, ch_out, expansion
        if expansion:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1
        self.block1 = Conv2D_BN_ReLU(ch_in, ch_out, kernel=3,
                                     padding=1, stride=stride1, bn=bn)
        self.block2 = Conv2D_BN_ReLU(ch_out, ch_out, kernel=3,
                                     padding=1, stride=stride2, bn=bn)

        self.shortcut = nn.Identity()
        if self.should_apply_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False, stride=stride1),
                BatchNorm2d_cond(self.ch_out, bn=bn)
            )

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = x + self.shortcut(input)
        return x

    @property
    def should_apply_shortcut(self):
        return self.ch_in != self.ch_out


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


### some UNets:
# UNet: original network. Will only train square-sized images. Pre-trained downsampling portion.
# UNet_V2: This version also works with any rectangular shaped RGB image.
# UNet_V3: with skip resnet connecting input to output.
class UNet_V4(nn.Module):
    """A UNet architecture with residual blocks. Not pretrained. This version allows to set BatchNorm to False.  \n
    See also https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet \n
    This version works with any rectangular shaped RGB image. . \n

    output: a predicted mask, not normalized (not last activation layer). So all loss functions must first use sigmoid activation.
    """

    def __init__(self, n_class, bn=False, SingleChannel = False):
        super().__init__()
        ch_in = 1 if SingleChannel else 3
        self.layer0 = nn.Sequential(Conv2D_BN_ReLU(ch_in, 64, kernel=7, padding=3, stride=2, bn=bn))  # base_layers[:3]
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2,
                                                 padding=1, dilation=1, ceil_mode=False),
                                    ResBlock(64, 64, bn=bn, expansion=False),
                                    ResBlock(64, 64, bn=bn, expansion=False),
                                    ) # base_layers[3:5]

        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = nn.Sequential(ResBlock(64, 128, bn=bn, expansion=True),
                                    ResBlock(128,128, bn=bn, expansion=False),
                                    )  # base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = nn.Sequential(ResBlock(128, 256, bn=bn, expansion=True),
                                    ResBlock(256, 256, bn=bn, expansion=False)
                                    ) # base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = nn.Sequential(ResBlock(256, 512, bn=bn, expansion=True),
                                    ResBlock(512, 512, bn=bn, expansion=False)
                                    ) # base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        #         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = upsample_UNet()

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1) # replace, the direct input - output link
        # self.conv_original_size1 = convrelu(64, 64, 3, 1) # replace, the direct input - output link
        self.ResNet_Input_to_Output = ResNet_fixedImgSize(bn=bn, SingleChannel=SingleChannel)  # 3 to 64
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)  # replace? Merging direct link with the upsampling.

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        # down sampling
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # up sampling
        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer4, layer3.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        layer2 = self.layer2_1x1(layer2)
        x = self.upsample(x, layer2.shape)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        layer1 = self.layer1_1x1(layer1)
        x = self.upsample(x, layer1.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        layer0 = self.layer0_1x1(layer0)
        x = self.upsample(x, layer0.shape)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        # Direct Resnet link, no downsampling.
        # x_original = self.conv_original_size0(input) # convrelu(3, 64, 3, 1)
        # x_original = self.conv_original_size1(x_original) # convrelu(64, 64, 3, 1)
        x_original = self.ResNet_Input_to_Output(input)  # 3 to 64

        x = self.upsample(x, x_original.shape)

        # linking the upsampled x with the input
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # Merging. convrelu(64 + 128, 64, 3, 1)

        out = self.conv_last(x)  # -> to size 1
        # maybe add a linear with ReLU here (https://discuss.pytorch.org/t/output-of-sigmoid-last-layer-of-cnn/39757)

        return out


class UNet_V3(nn.Module):
    """A UNet architecture with residual blocks. ResNet 18 with pretrained inital weight in downsampling portion. \n
    See also https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet \n
    This version works with any rectangular shaped RGB image. . \n

    output: a predicted mask, not normalized (not last activation layer). So all loss functions must first use sigmoid activation.
    """

    def __init__(self, n_class, bn = False, freeze=False):
        super().__init__()
        self.freeze = freeze
        base_model = models.resnet18() # by not saving it into class (self), you save a lot of memory.
        ModelPath = Path('/home/jupyter/PretrainedModels')
        machine_OS = platform.system()
        if machine_OS == 'Windows':
            resnetPath = r"C:\Users\M\.torch\models\resnet18-5c106cde.pth"
        elif machine_OS == 'Linux':
            resnetPath = ModelPath / 'resnet18-5c106cde.pth'

        base_model.load_state_dict(torch.load(resnetPath))
        base_layers = list(base_model.children())
        if self.freeze:
            with torch.no_grad():
                self.layer0 = nn.Sequential(*base_layers[:3])
        else:
            self.layer0 = nn.Sequential(*base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        if self.freeze:
            with torch.no_grad():
                self.layer1 = nn.Sequential(*base_layers[3:5])
        else:
            self.layer1 = nn.Sequential(*base_layers[3:5])

        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        #         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = upsample_UNet()

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1) # replace, the direct input - output link
        # self.conv_original_size1 = convrelu(64, 64, 3, 1) # replace, the direct input - output link
        self.ResNet_Input_to_Output = ResNet_fixedImgSize(bn=bn)  # 3 to 64
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)  # replace? Merging direct link with the upsampling.

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        # down sampling
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # up sampling
        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer4, layer3.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        layer2 = self.layer2_1x1(layer2)
        x = self.upsample(x, layer2.shape)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        layer1 = self.layer1_1x1(layer1)
        x = self.upsample(x, layer1.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        layer0 = self.layer0_1x1(layer0)
        x = self.upsample(x, layer0.shape)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        # Direct Resnet link, no downsampling.
        # x_original = self.conv_original_size0(input) # convrelu(3, 64, 3, 1)
        # x_original = self.conv_original_size1(x_original) # convrelu(64, 64, 3, 1)
        x_original = self.ResNet_Input_to_Output(input)  # 3 to 64

        x = self.upsample(x, x_original.shape)

        # linking the upsampled x with the input
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # Merging. convrelu(64 + 128, 64, 3, 1)

        out = self.conv_last(x)  # -> to size 1
        # maybe add a linear with ReLU here (https://discuss.pytorch.org/t/output-of-sigmoid-last-layer-of-cnn/39757)

        return out


class UNet_V2(nn.Module):
    """A UNet architecture with residual blocks. ResNet 18 with pretrained inital weight ni downsampling portion. \n
    See also https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet \n
    This version works with any rectangular shaped RGB image. . \n

    output: a predicted mask, not normalized (not last activation layer). So all loss functions must first use sigmoid activation.
    """

    def __init__(self, n_class, freeze=False):
        super().__init__()

        self.base_model = models.resnet18()
        ModelPath = Path('/home/jupyter/PretrainedModels')
        machine_OS = platform.system()
        if machine_OS == 'Windows':
            resnetPath = r"C:\Users\M\.torch\models\resnet18-5c106cde.pth"
        elif machine_OS == 'Linux':
            resnetPath = ModelPath / 'resnet18-5c106cde.pth'

        self.base_model.load_state_dict(torch.load(resnetPath))
        self.base_layers = list(self.base_model.children())
        if freeze:
            with torch.no_grad():
                self.layer0 = nn.Sequential(*self.base_layers[:3])
        else:
            self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        if freeze:
            with torch.no_grad():
                self.layer1 = nn.Sequential(*self.base_layers[3:5])
        else:
            self.layer1 = nn.Sequential(*self.base_layers[3:5])

        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        #         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = upsample_UNet()

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.activation = nn.Sigmoid()  # MHK added

    def forward(self, input):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        #         x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer4, layer3.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        #         x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = self.upsample(x, layer2.shape)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        #         x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = self.upsample(x, layer1.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        #         x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = self.upsample(x, layer0.shape)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        #         x = self.upsample(x)
        x = self.upsample(x, x_original.shape)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)  # -> to size 1
        # maybe add a linear with ReLU here (https://discuss.pytorch.org/t/output-of-sigmoid-last-layer-of-cnn/39757)

        return out


class upsample_UNet(nn.Module):
    def __init__(self, mode='bilinear'):
        # tensor_shape: from tensor.shape
        super().__init__()
        self.mode = mode

    def forward(self, input, tensor_shape):
        input.requires_grad_()
        m = nn.Upsample(size=(tensor_shape[2], tensor_shape[3]), mode=self.mode, align_corners=False)
        return m(input)


class UNet(nn.Module):
    """A UNet architecture with residual blocks. ResNet 18 with pretrained inital weight ni downsampling portion. \n
    See also https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet \n
    This version only worked with square images with a side length of 32*n, where n is an integer. \n
    The reason was that I used upsampling with factor 2, for 5 times.

    output: a predicted mask, not normalized (not last activation layer). So all loss functions must first use sigmoid activation.
    """

    def __init__(self, n_class, freeze=False):
        super().__init__()

        self.base_model = models.resnet18()
        ModelPath = Path('/home/jupyter/PretrainedModels')
        machine_OS = platform.system()
        if machine_OS == 'Windows':
            resnetPath = r"C:\Users\M\.torch\models\resnet18-5c106cde.pth"
        elif machine_OS == 'Linux':
            resnetPath = ModelPath / 'resnet18-5c106cde.pth'

        self.base_model.load_state_dict(torch.load(resnetPath))
        self.base_layers = list(self.base_model.children())
        if freeze:
            with torch.no_grad():
                self.layer0 = nn.Sequential(*self.base_layers[:3])
        else:
            self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        if freeze:
            with torch.no_grad():
                self.layer1 = nn.Sequential(*self.base_layers[3:5])
        else:
            self.layer1 = nn.Sequential(*self.base_layers[3:5])

        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.activation = nn.Sigmoid()  # MHK added

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)  # -> to size 1
        # maybe add a linear with ReLU here (https://discuss.pytorch.org/t/output-of-sigmoid-last-layer-of-cnn/39757)
        #         out_new = self.activation(out) # MHK added

        return out


def adjust_learning_rate(originalLR, optimizer, epoch, ratio=0.5, epochNumForDecrease=25):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
    Important lesson: rewriting an optimizer instead of just changing the lr means I erase the momentum!!!"""
    if np.mod(epoch, epochNumForDecrease) == 0:
        lr = originalLR * (ratio ** (epoch // epochNumForDecrease))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('New learning rate: ', lr)
