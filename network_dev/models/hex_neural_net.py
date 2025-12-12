import torch
import torch.nn as nn
import numpy as np

class InputConvBlock(nn.Module):
    '''Initial convolutional block to process input features.
    takes in (in_channels x 11 x 11) and (num_filters x 11 x 11) feature maps.'''
    # Padding on each side is 1, so stries by 1 across 13x13 input to maintain 11x11 size
    def __init__(self, in_channels, num_filters, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    
class ResidualBlock(nn.Module):
    '''Residual block with two convolutional layers, Batch norm and relu activation in between.'''

    def __init__(self, num_filters, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=padding),
            nn.BatchNorm2d(num_filters)
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        residual = X    # save input for skip connection
        out = self.layers(X)
        out = out + residual # skip connection to original input
        out = self.relu(out) # activation after addition 
        return out

class HexNeuralNet(nn.Module):
    def __init__(self, board_size=11, input_channels=3, num_filters=128):
        super(HexNeuralNet, self).__init__()
        
        # Shared Input Conv Block and Residual Tower
        self.input_conv = InputConvBlock(input_channels, num_filters=num_filters)
        self.residual_tower = nn.ModuleList([
            ResidualBlock(num_filters, kernel_size=3) for _ in range(20)])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),               # flattens from 2 x 11 x 11 to 242 
            nn.Linear(2 * board_size * board_size, board_size * board_size), # from 242 to 121
            nn.Softmax(dim=1)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),               
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, X):
        out = self.input_conv(X)
        for residual_block in self.residual_tower:
            out = residual_block(out)

        # For probabilities 
        out_policy = self.policy_head(out)
        
        # For value 
        out_value = self.value_head(out)

        return out_policy, out_value

    def predict(self, state):
        """
        state shape: (3, 11, 11)
        Returns:
            policy_probs: (121,) numpy array
            value: scalar
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            policy_probs, value = self.forward(state_tensor)
            return policy_probs.squeeze(0).numpy(), value.item()
        
# Test the HexNeuralNet
model = HexNeuralNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
probs, value = model(torch.randn(1, 3, 11, 11))
print("Policy logits shape:", probs.shape)  # Expected: (1, 121)
print("Value shape:", value.shape)          # Expected: (1, 1)