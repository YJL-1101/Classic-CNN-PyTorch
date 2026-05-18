import torch
from torch import nn
from torchsummary import summary

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(25088,4096)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(4096,4096)
        self.f3 = nn.Linear(4096,10)


        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088,4096),
            nn.ReLU(),
            self.dropout,
            nn.Linear(4096,4096),
            nn.ReLU(),
            self.dropout,
            nn.Linear(4096,10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 同样使用 kaiming_normal_，自动根据输入维度计算合适的标准差
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.f3(x)

        return x

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)
    print(summary(model, (3, 224, 224)))