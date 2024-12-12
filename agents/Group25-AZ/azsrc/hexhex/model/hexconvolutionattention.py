import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        # Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // 8, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_attention_scale = 0.042
        self.spatial_attention_scale = 0.0625

        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(F.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.global_max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)

        x = x * (1 + self.channel_attention_scale * (channel_attention - 1))

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

        x = x * (1 + self.spatial_attention_scale * (spatial_attention - 1))

        return x


class SkipLayerBias(nn.Module):
    def __init__(self, channels, reach, scale=1):
        super(SkipLayerBias, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.attention = AttentionBlock(channels)
        self.scale = scale

    def forward(self, x):
        residual = x
        x = self.bn(self.conv(x))
        x = self.attention(x)  # Apply attention
        return swish(residual + self.scale * x)

    def freeze(self):
        """Freeze all parameters in the layer"""
        # Freeze conv layer
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # Freeze batch norm layer
        for param in self.bn.parameters():
            param.requires_grad = False
            
        # Freeze attention block
        for param in self.attention.parameters():
            param.requires_grad = False


class Conv(nn.Module):
    '''
    model consists of a convolutional layer to change the number of channels from two input channels to intermediate channels
    then a specified amount of residual or skip-layers https://en.wikipedia.org/wiki/Residual_neural_network
    then policyconv reduce the intermediate channels to one
    value range is (-inf, inf) 
    for training the sigmoid is taken, interpretable as probability to win the game when making this move
    for data generation and evaluation the softmax is taken to select a move
    '''
    def __init__(self, board_size, layers, intermediate_channels, reach, export_mode):
        super(Conv, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(2, intermediate_channels, kernel_size=2*reach+1, padding=reach-1)
        self.skiplayers = nn.ModuleList([SkipLayerBias(intermediate_channels, 1) for _ in range(layers)])
        # for idx in range(layers):
        #     skip_layer = SkipLayerBias(intermediate_channels, 1)
        #     skip_layer.freeze()  # Freeze the layer immediately after creation
        #     self.skiplayers.append(skip_layer)
        self.policyconv = nn.Conv2d(intermediate_channels, 1, kernel_size=2*reach+1, padding=reach, bias=False)
        self.bias = nn.Parameter(torch.zeros(board_size**2))
        self.export_mode = export_mode

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        if self.export_mode:
            return self.policyconv(x).view(-1, self.board_size ** 2) + self.bias
        #  illegal moves are given a huge negative bias, so they are never selected for play
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        return self.policyconv(x).view(-1, self.board_size**2) + self.bias - illegal


class RandomModel(nn.Module):
    '''
    outputs negative values for every illegal move, 0 otherwise
    only makes completely random moves if temperature*temperature_decay > 0
    '''
    def __init__(self, board_size):
        super(RandomModel, self).__init__()
        self.board_size = board_size

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        return torch.rand_like(illegal) - illegal


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