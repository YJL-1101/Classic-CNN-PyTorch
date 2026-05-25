from contextlib import nullcontext

import torch
import torch.nn as nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,input_channel,output_channel,strides=1,flag=False,):
        super(Residual,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=3,padding=1,stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        if flag:
            self.conv3 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=strides)
        else:
            self.conv3 = None

    def forward(self,x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)

        y = x+y
        y = self.relu(y)

        return y

class ResNet18(nn.Module):
    def __init__(self,Residual):
        super(ResNet18,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3,2,1)
        )
        self.block2 = nn.Sequential(
            Residual(64,64),
            Residual(64,64)
        )

        self.block3 = nn.Sequential(
            Residual(64,128,2,True),
            Residual(128,128)
        )

        self.block4 = nn.Sequential(
            Residual(128, 256, 2, True),
            Residual(256, 256)
        )

        self.block5 = nn.Sequential(
            Residual(256, 512, 2, True),
            Residual(512, 512)
        )

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model,(3,224,224)))