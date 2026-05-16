import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(3,96,11,4)
        self.p2 = nn.MaxPool2d(3,2)
        self.c3 = nn.Conv2d(96,256,5,1,2)
        self.p4 = nn.MaxPool2d(3,2)
        self.c5 = nn.Conv2d(256,384,3,1,1)
        self.c6 = nn.Conv2d(384,384,3,1,1)
        self.c7 = nn.Conv2d(384,256,3,1,1)
        self.p8 = nn.MaxPool2d(3,2)

        self.flatten = nn.Flatten()

        self.f1 = nn.Linear(6*6*256,4096)
        self.f2 = nn.Linear(4096,4096)
        self.f3 = nn.Linear(4096,10)

    def forward(self,x):
        #卷积部分
        x = self.c1(x)
        x = self.relu(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.relu(x)
        x = self.p4(x)
        x = self.c5(x)
        x = self.relu(x)
        x = self.c6(x)
        x = self.relu(x)
        x = self.c7(x)
        x = self.relu(x)
        x = self.p8(x)
        #平展部分
        x = self.flatten(x)
        #全连接部分
        x = self.relu(self.f1(x))
        #x = F.dropout(x, p=0.5)
        x = self.dropout(x)
        x = self.relu(self.f2(x))
        #x = F.dropout(x, p=0.5)
        x = self.dropout(x)
        x = self.f3(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (3,227,227)))

