import torch
import torch.nn as nn

class BTCPriceClassifier(nn.Module):
    """
    Feedforward Neural Network for binary classification of BTC price direction.
    Input: 5 engineered features
    Output: Single probability [0, 1] representing likelihood of price moving UP.
    """
    def __init__(self, input_dim=5):
        super(BTCPriceClassifier, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: Expand 5 features to 32 hidden neurons
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),  # Normalizes activations across batches for stability
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Randomly drops 20% of neurons to stop overfitting
            
            # Layer 2: Compress from 32 to 16 neurons
            nn.Linear(32, 16),
            nn.ReLU(),

            # Layer 3: Output layer (1 neuron)
            nn.Linear(16, 1),
            nn.Sigmoid()    # Squashes output into a probability between 0 and 1
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        return self.network(x)

if __name__ == "__main__":
    # Smoke test: test the model with a fake batch of 3 samples
    test_model = BTCPriceClassifier(input_dim=5)
    print("Model Architecture:\n", test_model)
    # Fake input: 3 samples, 5 features each
    dummy_input = torch.randn(3, 5)
    output = test_model(dummy_input)
    print("\nDummy Input shape:", dummy_input.shape)
    print("Model Output (Probabilities):\n", output)