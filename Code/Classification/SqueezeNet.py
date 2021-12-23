from torchsummary import summary
import torch.nn as nn
import warnings
import torch
# from torch.autograd import Variable

warnings.filterwarnings("ignore")

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.LeakyReLU(inplace=True)

        # using MSR initilization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=9):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False) # 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = fire(32, 16, 32)
        self.fire3 = fire(64, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(2560, 384)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(384,128)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128,num_classes)

        self.softmax = nn.LogSoftmax(dim=1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)

        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

def squeezenet(pretrained=False):
    model = SqueezeNet(1, 9)
    # model2 = SqueezeNet(2, 9)
    # model3 = SqueezeNet(3, 9)
    # x1 = torch.randn(3, 1, 512, 512)
    # x2 = torch.randn(3, 2, 512, 512)
    # x3 = torch.randn(3, 3, 512, 512)
    # print("Ouput 1: ",model1(x1).shape)
    # print("Ouput 2: ",model2(x2).shape)
    # print("Ouput 3: ",model3(x3).shape)

    # summary(model, x1)

    # inp = Variable(torch.randn(3,1,512,512))
    # out = model.forward(inp)
    # print(out.size())
    # print(model)
    
    ## Save model's structure
    # torch.save(model.state_dict(), './runs/Classification/SqueezeNet/SqueezeNet.pth')

    return model

if __name__ == '__main__':
    model = squeezenet()
    summary(model, (1, 512, 512))
