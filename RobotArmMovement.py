import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class RobotArmMovement(nn.Module):
  def __init__(self, max_seq_len, action_dim=4):
    super(RobotArmMovement, self).__init__()
    self.max_seq_len = max_seq_len
    self.action_dim = action_dim
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

    )

    self.distance_fc = nn.Sequential(
      nn.Linear(1,16),
      nn.ReLU()
    )

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 8 * 8 + 16, 256),
      nn.ReLU(),
      nn.Linear(256, ),
    )

  def forward(self, image, distance):
    cnn_features = self.cnn(image)

    distance_features = self.distance_fc(distance)

    combined_features = torch.cat((cnn_features, distance_features), dim=1)

    output = self.fc(combined_features)
    return output.view(-1, self.max_seq_len, self.action_dim)

    

    