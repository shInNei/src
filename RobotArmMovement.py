import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class RobotArmMovement(nn.Module):
  def __init__(self, max_seq_len=14, action_dim=4):
    super(RobotArmMovement, self).__init__()
    self.max_seq_len = max_seq_len
    self.action_dim = action_dim
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((3, 3)),

    )

    self.distance_fc = nn.Sequential(
      nn.Linear(1,16),
      nn.ReLU()
    )

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 3 * 3 + 16, 128),
      nn.ReLU(),
      nn.Linear(128, self.max_seq_len * self.action_dim),
    )


  # def clip_output(self, output):
  #   """Clamp the output values to ensure they stay within a valid range."""
  #   return torch.clamp(output, min=0, max=1)

  def create_mask(self, joint_angles_tensor, padding_value=0):
    # Create a mask: 1 where joint angles are not equal to padding value, 0 where they are
    mask = (joint_angles_tensor >= padding_value).float()  # Convert to float for element-wise multiplication
    return mask

  def forward(self, image, distance):
    cnn_features = self.cnn(image)
    cnn_features = cnn_features.view(cnn_features.size(0), -1)

    distance_features = self.distance_fc(distance)

    combined_features = torch.cat((cnn_features, distance_features), dim=1)

    output = self.fc(combined_features)
    output = output.view(-1, self.max_seq_len, self.action_dim)
    # output = self.clip_output(output)
    return output

    

    