import torch
import torch.nn as nn
import math
from collections import OrderedDict

__all__ = ['mobilenetv2']


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.ReLU6(inplace=True))


def pointwise_conv_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Mobilenetv2(nn.Module):
    def __init__(self, width_mult=1.):
        super(Mobilenetv2, self).__init__()
        self.input_channel = 32
        last_channel = 1280
        self.width_mult = width_mult
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # building first layer
        self.layer0 = nn.Sequential(conv_bn(3, self.input_channel, 2))

        # building inverted residual blocks
        self.layer1 = self._make_layer(1, 16, 1, 1)
        self.layer2 = self._make_layer(6, 24, 2, 2)

        # the 52*52 feature map output
        self.layer3 = self._make_layer(6, 32, 3, 2)
        self.layer3_output = pointwise_conv_bn(self.input_channel, 128)

        # the 26*26 feature map output
        self.layer4 = self._make_layer(6, 64, 4, 2)
        self.layer5 = self._make_layer(6, 96, 3, 1)
        self.layer5_output = pointwise_conv_bn(self.input_channel, 384)

        self.layer6 = self._make_layer(6, 160, 3, 2)
        self.layer7 = self._make_layer(6, 320, 1, 1)

        self.last_layer = pointwise_conv_bn(self.input_channel, self.last_channel)

        # the demensions of output feature map channels
        self.layers_out_filters = [128, 384, 1280]
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, t, c, n, s):
        layers = []
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                layers.append(("IRB_{}".format(i), InvertedResidual(self.input_channel, output_channel, s, expand_ratio=t)))
            else:
                layers.append(("IRB_{}".format(i), InvertedResidual(self.input_channel, output_channel, 1, expand_ratio=t)))
            self.input_channel = output_channel
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out52 = self.layer3_output(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out26 = self.layer5_output(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out13 = self.last_layer(x)
        return out52, out26, out13


def mobilenetv2(pretrained, **kwargs):
    """Constructs a mobilenetv2 model.
    """
    model = Mobilenetv2()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    config = {"model_params": {"backbone_name": "darknet_53"}}
    m = Mobilenetv2()
    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())