from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
from model import LeNet

def test_data_process():
    # 加载测试数据集
    data = FashionMNIST(root="./data",
                        train=False,
                        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                        download=True)


    test_data_loader = Data.DataLoader(dataset=data,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0)
    return test_data_loader

def test_model_process(model,test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #模型放入设备中
    model = model.to(device)

    # 初始化参数
    # 准确个数
    test_ac_num = 0.0
    # 测试集样本数量
    test_num = 0
    #不计算梯度
    with torch.no_grad():
        for test_data_x,test_data_y in test_data_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            #前向传播
            output = model(test_data_x)

            pre_lab = torch.argmax(output,dim=1)

            test_ac_num += torch.sum(pre_lab == test_data_y)

            test_num += test_data_x.size(0)

    test_acc = test_ac_num.double().item() / test_num
    print("测试准确率：",test_acc)

#测试过程中打印单张图的测试结果
def test_model_process_v2(model,test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #模型放入设备中
    model = model.to(device)

    #不计算梯度
    classes = ['T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot']
    with torch.no_grad():
        for test_data_x,test_data_y in test_data_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            #前向传播
            output = model(test_data_x)

            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab.item()
            label = test_data_y.item()
            print("预测值：",classes[result],"   标签值：",classes[label])




if __name__=="__main__":
    model = LeNet()

    model.load_state_dict(torch.load("best_model.pth"))

    test_data_loader = test_data_process()

    #test_model_process(model,test_data_loader)
    test_model_process_v2(model, test_data_loader)

