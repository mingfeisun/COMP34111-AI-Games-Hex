import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        input_dim = 3 * 121

        # Policy head: outputs logits for 121 moves
        self.policy_head = nn.Linear(input_dim, 121)
        self.softmax = nn.Softmax(dim=1)

        # Value head: outputs a scalar in [-1, 1]
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x shape: (batch_size, 3, 11, 11)
        Returns:
            policy_logits: (batch_size, 121)
            value: (batch_size, 1)
        """

        # Flatten
        x = x.view(x.size(0), -1)

        # Compute policy logits and value
        policy_logits = self.policy_head(x) 
        policy_probs = self.softmax(policy_logits)
        value = torch.tanh(self.value_head(x)) 

        return policy_probs, value

    def predict(self, state):
        """
        state shape: (3, 11, 11)
        Returns:
            policy_probs: (121,) numpy array
            value: scalar
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            policy_probs, value = self.forward(state_tensor)
            return policy_probs.squeeze(0).numpy(), value.item()

model = DummyModel()
probs, value = model(torch.randn(1, 3, 11, 11))
print("Policy logits shape:", probs.shape)  # Expected: (1, 121)
print("Value shape:", value.shape)          # Expected: (1, 1)