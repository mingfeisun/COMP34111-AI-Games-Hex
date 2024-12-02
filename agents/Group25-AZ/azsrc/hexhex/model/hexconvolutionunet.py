import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2D -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Attention block for refining feature maps in the skip connections."""
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, intermediate_channels, layers, board_size, export_mode):
        super(UNetWithAttention, self).__init__()
        self.board_size = board_size
        self.export_mode = export_mode

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder with attention
        self.att4 = AttentionBlock(1024, 512, 256)
        self.dec4 = ConvBlock(1024, 512)

        self.att3 = AttentionBlock(512, 256, 128)
        self.dec3 = ConvBlock(512, 256)

        self.att2 = AttentionBlock(256, 128, 64)
        self.dec2 = ConvBlock(256, 128)

        self.att1 = AttentionBlock(128, 64, 32)
        self.dec1 = ConvBlock(128, 64)

        # Final output layers
        self.policyconv = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(board_size ** 2))
    
    def forward(self, x):
        # Compute x_sum
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1, self.board_size ** 2)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with attention
        att4 = self.att4(bottleneck, enc4)
        dec4 = self.dec4(torch.cat([bottleneck, att4], dim=1))

        att3 = self.att3(dec4, enc3)
        dec3 = self.dec3(torch.cat([dec4, att3], dim=1))

        att2 = self.att2(dec3, enc2)
        dec2 = self.dec2(torch.cat([dec3, att2], dim=1))

        att1 = self.att1(dec2, enc1)
        dec1 = self.dec1(torch.cat([dec2, att1], dim=1))

        # Policy convolution
        x = self.policyconv(dec1).view(-1, self.board_size ** 2)

        if self.export_mode:
            return x + self.bias
        
        # Illegal moves handling
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1) - 1) * 1000) * 10).unsqueeze(1).expand_as(x_sum) - x_sum
        return x + self.bias - illegal
#
class NoSwitchWrapperModel(nn.Module):
    '''
    same functionality as parent model, but switching is illegal
    '''
    def __init__(self, model):
        super(NoSwitchWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model

    def forward(self, x):
        illegal = 1000*torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        return self.internal_model(x)-illegal
    
class RotationWrapperModel(nn.Module):
    '''
    evaluates input and its 180° rotation with parent model
    averages both predictions
    '''
    def __init__(self, model, export_mode):
        super(RotationWrapperModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model
        self.export_mode = export_mode

    def forward(self, x):
        if self.export_mode:
            return self.internal_model(x)
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y)/2