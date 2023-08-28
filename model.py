import torch
from torch import nn
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.model(x)
    
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        features = [64,128,256,512]
        self.encoder = nn.ModuleList()
        in_channels = 3 # Accounts for rgb exclude a
        for out_channels in features:
            self.encoder.append(DoubleConv(in_channels,out_channels))
            in_channels = out_channels
        
        self.decoder = nn.ModuleList()
        in_channels = 1024
        for out_channels in reversed(features):
            self.decoder.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleNeck = DoubleConv(512,1024)

        self.transpose = nn.ModuleList()
        in_channels = 1024
        for out_channels in reversed(features):
            self.transpose.append(nn.ConvTranspose2d(in_channels, out_channels, (2,2), 2))
            in_channels = out_channels

        self.finalLayer = DoubleConv(64, 1) # 1 channl outputted
    def forward(self, x):
        skipConnect = []
        for convLayer in self.encoder:
            x = convLayer(x)
            skipConnect.append(x)
            x = self.maxPool(x)

        x = self.bottleNeck(x)

        for skipValue, upSample, convLayer in zip(reversed(skipConnect), self.transpose, self.decoder):
            x = upSample(x)
            x = torch.cat((skipValue, x), dim=1)
            x = convLayer(x)
        
        return self.finalLayer(x)



if __name__ == "__main__":
    #create 2d array

    x = torch.randn((3,3,160,160))
    model = UNET()
    result = model(x)
    print(result.shape)
        

# 64, 3, 3, 3 # in channels is 2
# [64, 4, 3, 3] 