import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False)  # stride=2 reduces size
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.Conv2d(in_channel, out_channel, 1, 2, 0, bias=False)

    def forward(self, x):
        res = x
        res = self.pool(res)        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = x + res        
        x = self.relu1(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.block1 = ResBlock(64, 128)
        self.block2 = ResBlock(128, 256)
        self.block3 = ResBlock(256, 512)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x
