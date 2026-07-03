import torch
import torch.nn as nn
import torchvision.models as models

class StateBCPolicy(nn.Module):
    """
    Option A: Fast State-Based Policy
    Takes the raw 6D physics observation and outputs 2D action.
    """
    def __init__(self):
        super().__init__()
        
        # A simple 3-layer Multi-Layer Perceptron (MLP)
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output layer for the 2 actions
        )
        
    def forward(self, obs):
        return self.net(obs)


class VisualBCPolicy(nn.Module):
    """
    Option B: Visual-Based Policy
    Takes a (3, 224, 224) image + (2) Target Position and outputs 2D action.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Vision Encoder: ResNet-18
        # We load a standard ResNet-18 but remove the final classification layer
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # Output is (Batch, 512, 1, 1)
        self.encoder_dim = 512
        
        # 2. Policy Head (MLP)
        # Input to MLP is the 512 image features + 2 target coordinates = 514
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output layer for the 2 actions
        )
        
    def forward(self, img, target_pos):
        # 1. Extract image features
        img_features = self.encoder(img) # Shape: (Batch, 512, 1, 1)
        img_features = img_features.view(img_features.size(0), -1) # Flatten to (Batch, 512)
        
        # 2. Concatenate the target position
        concat_features = torch.cat([img_features, target_pos], dim=1) # Shape: (Batch, 514)
        
        # 3. Predict action
        action = self.mlp(concat_features) # Shape: (Batch, 2)
        
        return action

if __name__ == "__main__":
    # Test State Model
    state_model = StateBCPolicy()
    dummy_obs = torch.randn(1, 6) # Batch Size 1, 6 features
    state_out = state_model(dummy_obs)
    print(f"State Model Output Shape: {state_out.shape} (Should be 1, 2)")
    
    # Test Visual Model
    visual_model = VisualBCPolicy()
    dummy_img = torch.randn(1, 3, 224, 224) # Batch Size 1, 3 Channels, 224x224
    dummy_target = torch.randn(1, 2)        # Batch Size 1, 2 Target Coords
    vis_out = visual_model(dummy_img, dummy_target)
    print(f"Visual Model Output Shape: {vis_out.shape} (Should be 1, 2)")
