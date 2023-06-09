import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.up(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv(x)
        return x
    
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = double_conv(in_channels, out_channels)
        self.conv2 = double_conv(out_channels, out_channels)
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UNet(nn.Module):
    def __init__(self, n_channels, n_class, depth):
        super().__init__()
        
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # create layers
        for i in range(depth):
            ins = n_channels if i == 0 else outs
            outs = 64 * 2**i
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_last = nn.Conv2d(outs, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoder_outs = []

        # encoder part
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # decoder part
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        x = self.conv_last(x)
        return self.sigmoid(x)
